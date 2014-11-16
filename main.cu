#include<stdio.h>
#include<stdint.h>

#include"support.h"
#include"kernel.cu"

int main(int argc, char **argv) {
  Timer timer;

  printf("\nSetting up the problem...");

  float *Temp_h;
  float *Enrg_h;
  float *Magn_h;
  float *latt_h;
  float *Temp_d;
  float *Enrg_d;
  float *Magn_d;
  float *latt_d;
  int num_temps, latt_len, num_steps;
  cudaError_t cuda_ret;

  if(argc < 4) {
    printf("\n\tUsage: %s num_temps latt_len num_steps\n", argv[0]);
    exit(0);
  }

  num_temps = atoi(argv[1]);
  latt_len = atoi(argv[2]); 
  num_steps = atoi(argv[3]);

  printf("Allocating host variables..."); fflush(stdout);
  startTime(&timer);

  Temp_h = (float *)malloc(num_temps*sizeof(float));
  Enrg_h = (float *)malloc(num_temps*sizeof(float));
  Magn_h = (float *)malloc(num_temps*sizeof(float));
  latt_h = (float *)malloc(latt_len*sizeof(float));

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  init_temp(Temp_h, num_temps);
	
  printf("Allocating device variables..."); fflush(stdout);
  startTime(&timer);

  cuda_ret = cudaMalloc((void **)&Temp_d, num_temps*sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&Enrg_d, num_temps*sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&Magn_d, num_temps*sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&latt_d, latt_len*sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
	
  cudaDeviceSynchronize();
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  //cuda_ret = cudaMemcpy(Temp_d, Temp_H, num_temps*sizeof(float), cudaMemcpyHostToDevice);

  cuda_ret = cudaMemset(Enrg_d, 0, num_temps*sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to zero energies");

  for(int i = 0; i < num_temps; i++) {
    init_latt(latt_h, latt_len);
    cuda_ret = cudaMemcpy(latt_d, latt_h, latt_len*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy lattice to device");
    cudaDeviceSynchronize();
    find_xy_parameters(Temp_h[i], latt_d, latt_len, num_steps, Enrg_d+i, Magn_d+i);
  }

  cuda_ret = cudaMemcpy(Enrg_h, Enrg_d, num_temps*sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy results from device");
  cuda_ret = cudaMemcpy(Magn_h, Magn_d, num_temps*sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy results from device");
  cudaDeviceSynchronize();

  write_data(Temp_h, Enrg_h, Magn_h, num_temps);

  free(Temp_h);
  free(Enrg_h);
  free(Magn_h);
  free(latt_h);

  cudaFree(Temp_d);
  cudaFree(Enrg_d);
  cudaFree(Magn_d);
  cudaFree(latt_d);

  return 0;
}
