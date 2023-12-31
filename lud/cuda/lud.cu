#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>
#include <cuda.h>
#include <cusolverDn.h>

#include "common.h"

#ifdef RD_WG_SIZE_0_0
    #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
    #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
    #define BLOCK_SIZE RD_WG_SIZE
#else
    #define BLOCK_SIZE 16
#endif

extern void
lud_cuda(double *d_m, int matrix_dim);

static int do_verify = 0, use_lib = 1;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {"rodina", 0, NULL, 'r'},
  {0,0,0,0}
};

#define CHECK_CUDA(call)                                          \
  if ((call) != cudaSuccess)                                      \
  {                                                               \
    fprintf(stderr, "CUDA error at %s %d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;                                          \
  }

#define CHECK_CUSOLVER(call)                                          \
  if ((call) != CUSOLVER_STATUS_SUCCESS)                              \
  {                                                                   \
    fprintf(stderr, "cuSOLVER error at %s %d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;                                              \
  }

int main(int argc, char *argv[])
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
  cusolverDnHandle_t handle;
  int *devIpiv, *devInfo;
  int matrix_dim = 32; // example matrix size
  int Lwork = 0;
  int opt, option_index = 0;
  func_ret_t ret;
  const char *input_file = NULL;
  double *m, *d_m, *mm;
  double *devWork = NULL;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:",
                            long_options, &option_index)) != -1)
  {
    switch (opt)
    {
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 'r':
      use_lib = 0;
      break;
    case 's':
      matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
              argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if ((optind < argc) || (optind == 1))
  {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file)
  {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS)
    {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  }
  else if (matrix_dim)
  {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS)
    {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }

  else
  {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }
  if (do_verify)
  {
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  printf("Rodina flag: %d\n", use_lib);

  // Allocate the device matrix
  CHECK_CUDA(cudaMalloc((void **)&d_m, matrix_dim * matrix_dim * sizeof(double)));
  printf("Performing LU decomposition\n");

  // Copy the host matrix to the device
  CHECK_CUDA(cudaMemcpy(d_m, m, matrix_dim * matrix_dim * sizeof(double), cudaMemcpyHostToDevice));

  
  if(use_lib == 0){
    stopwatch_start(&sw);
    lud_cuda(d_m, matrix_dim);
    stopwatch_stop(&sw);
  }
  else {
    stopwatch_start(&sw);
    // Create the cuSOLVER handle
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    // Allocate the pivot array and info parameter on the device
    CHECK_CUDA(cudaMalloc((void **)&devIpiv, matrix_dim * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&devInfo, sizeof(int)));

    // Compute the LU decomposition
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(handle, matrix_dim, matrix_dim, d_m, matrix_dim, &Lwork));
    CHECK_CUDA(cudaMalloc((void **)&devWork, sizeof(double) * Lwork));
    CHECK_CUSOLVER(cusolverDnDgetrf(handle, matrix_dim, matrix_dim, d_m, matrix_dim, devWork, devIpiv, devInfo));
    stopwatch_stop(&sw);
  }
  
  printf("Time consumed without memcpy(ms): %lf\n", 1000*get_interval_by_sec(&sw));

  // Copy the result back to the host
  CHECK_CUDA(cudaMemcpy(m, d_m, matrix_dim * matrix_dim * sizeof(double), cudaMemcpyDeviceToHost));

  printf("LU decomposition completed\n");

  if(use_lib == 1) {
    int hostInfo;
    CHECK_CUDA(cudaMemcpy(&hostInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Devinfo: %d\n", hostInfo);
  }


  if (do_verify){
    printf("After LUD\n");
    //print_matrix(m, matrix_dim);
    //print_matrix(mm, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim, use_lib);
    free(mm);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_m));
  if(use_lib == 1) {
    CHECK_CUDA(cudaFree(devIpiv));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(devWork));
    cusolverDnDestroy(handle);
  }
  free(m);

  return 0;
}