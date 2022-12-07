#include <stdio.h>

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("device: %d\n", i);
    printf("  name: %s\n", prop.name);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  total global memory: %ld MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("  max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  max thread dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  max grid size: (%ld, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n");
  }
}