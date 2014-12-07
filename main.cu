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
  float **latt_arr_h;
  float *Temp_d;
  float *Enrg_d;
  float *Magn_d;
  float **latt_arr_d;
  int num_temps, latt_len, num_steps;
  cudaError_t cuda_ret;

  if(argc < 4) {
    printf("\n\tUsage: %s num_temps latt_len num_steps\n", argv[0]);
    exit(0);
  }

  num_temps = atoi(argv[1]);
  latt_len = atoi(argv[2]); 
  num_steps = atoi(argv[3]);
	printf("Running with num_temps %i, latt_len %i and num_steps %i\n", num_temps, latt_len, num_steps);

  printf("Allocating host variables..."); fflush(stdout);
  startTime(&timer);

  Temp_h = (float *)malloc(num_temps*sizeof(float));
  Enrg_h = (float *)malloc(num_temps*sizeof(float));
  Magn_h = (float *)malloc(num_temps*sizeof(float));
  latt_arr_h = (float **)malloc(num_temps*sizeof(float *));

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
	cuda_ret = cudaMalloc((void **)&latt_arr_d, num_temps*sizeof(float *));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
	for(int i = 0; i < num_temps; i++) {
		cuda_ret = cudaMalloc((void **)latt_arr_h+i, latt_len*sizeof(float));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
		cudaMemset(latt_arr_h[i], 0, latt_len*sizeof(float));
	}
	
  cudaDeviceSynchronize();
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  //cuda_ret = cudaMemcpy(Temp_d, Temp_H, num_temps*sizeof(float), cudaMemcpyHostToDevice);

  cuda_ret = cudaMemset(Enrg_d, 0, num_temps*sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to zero energies");

	cuda_ret = cudaMemcpy(latt_arr_d, latt_arr_h, num_temps*sizeof(float *), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy from host to device");

	printf("Running trials..."); fflush(stdout);
	startTime(&timer);

	find_xy_parameters(num_temps, latt_arr_d, latt_len, num_steps, Enrg_d, Magn_d);

	cudaDeviceSynchronize();
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	printf("Copying from device..");
	startTime(&timer);
  cuda_ret = cudaMemcpy(Enrg_h, Enrg_d, num_temps*sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy results from device");
  cuda_ret = cudaMemcpy(Magn_h, Magn_d, num_temps*sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy results from device");
  cudaDeviceSynchronize();
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
	for(int i=0; i < num_temps; i++) {
		float *dev_ptr = latt_arr_h[i];
		/*
		latt_arr_h[i] = (float *)malloc(latt_len*sizeof(float));
		cuda_ret = cudaMemcpy(latt_arr_h[i], dev_ptr, latt_len*sizeof(float), cudaMemcpyDeviceToHost);
		if(cuda_ret != cudaSuccess) FATAL("Unable to cpy results from device");
		printf("lattice %i\n", i);
		for(int j=0; j < latt_len; j++) printf("%f\n", latt_arr_h[i][j]);
		free(latt_arr_h[i]);
		*/
		cudaFree(dev_ptr);
	}

  write_data(Temp_h, Enrg_h, Magn_h, num_temps);

  free(Temp_h);
  free(Enrg_h);
  free(Magn_h);
  free(latt_arr_h);

  cudaFree(Temp_d);
  cudaFree(Enrg_d);
  cudaFree(Magn_d);
  cudaFree(latt_arr_d);

  return 0;
}
