#include"support.h"
#include<cuda.h>
#include<curand_kernel.h>
#include<time.h>

#define norm_vect_len(m) (sqrt(pow((m).x, 2) + pow((m).y, 2))/(m).num_elem)
#define BLOCK_SIZE 1024

typedef struct {
  float x;
  float y;
  int num_elem;
} magn_t;


__device__ float global_temp[BLOCK_SIZE];
__global__ void calc_energy(float **latt_arr, unsigned int latt_len, float *nrg, int ind) {
  unsigned int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ float local_temp[BLOCK_SIZE/2];
  if(threadIdx.x < BLOCK_SIZE/2) local_temp[threadIdx.x] = 0.0;

	float *latt = latt_arr[ind];

  if(index >= latt_len) return;
	
  float retval = 2.0;
  float spin = latt[index];
	
  for(int i = 0; i < latt_len; i++)
    retval -= 2*__cosf(latt[i] - spin);
  retval /= latt_len;
	if(threadIdx.x > BLOCK_SIZE/2) local_temp[threadIdx.x-BLOCK_SIZE/2] = retval;
  __syncthreads();
  for(int stride = 2; stride < BLOCK_SIZE; stride <<= 1) {
    if(index < BLOCK_SIZE/stride) 
      retval += local_temp[threadIdx.x];
    __syncthreads();
    if(index >= BLOCK_SIZE/(stride<<1) && index < BLOCK_SIZE/stride)
      local_temp[threadIdx.x-(BLOCK_SIZE/(stride<<1))] = retval;
    __syncthreads();
  }
	
  if(threadIdx.x == 0) {
		global_temp[blockIdx.x] = retval;
	}
  __syncthreads();
	
  if(index >= gridDim.x) return;
  for(int stride = 2; stride < gridDim.x; stride <<= 1) {
    if(index % stride == 0)
      global_temp[index] += global_temp[index + (stride >> 1)];
  }
  __syncthreads();
	
  if(index == 0) {
		*nrg = global_temp[0];
	}

}
__device__ float calc_delta_nrg(float new_val, float old_val, float *latt, int latt_len) {
	float delta_nrg = 2.*__cosf(new_val-old_val)-2.;
	for(int j = 0; j < latt_len; j++) {
		delta_nrg -= 2*__cosf(new_val-latt[j])-2*__cosf(old_val-latt[j]);
	}
	delta_nrg /= latt_len;
	return delta_nrg;
}
__device__ void perturb_latt(float *latt, int latt_len, int num_steps, float temp, int index, curandState s) {
		float old_val = latt[index];
		for(int i = 0; i < num_steps; i++) {
			float new_val = curand_uniform(&s) * PI_2;
			float delta_nrg = calc_delta_nrg(new_val, old_val, latt, latt_len);
			if((delta_nrg < 0) && (curand_uniform(&s) < exp(-delta_nrg/temp))) {
				//latt[index] = new_val;
				old_val = new_val;
			}
		}
		latt[index] = old_val;
}
__global__ void iterate_nrg(int num_temps, float **latt_arr, unsigned int latt_len, int num_steps, curandState *states, float *Enrg) {
	int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(index >= latt_len) return;
  //__shared__ float temp_latt[BLOCK_SIZE];
  curandState s = states[index];
  curand_init(1234, index, 0, &s);
	
	for(int i = 0; i < num_temps; i++) {
		latt_arr[i][index] = curand_uniform(&s) * PI_2;
	}
	
	__syncthreads();
	
	float temp = 4.0;
	int temp_i = 0;
	for(float temp = MIN_TEMP; temp < MAX_TEMP; temp += TEMP_DIFF/num_temps, temp_i++) {
		perturb_latt(latt_arr[temp_i], latt_len, num_steps, temp, index, s);
	}
}


			/*
				for(int off = 0; off < latt_len; off += BLOCK_SIZE) {
				temp_latt[threadIdx.x] = latt[off+threadIdx.x];
				__syncthreads();
				for(int j = 0; j < BLOCK_SIZE; j++)
        delta_nrg -= 2*__cosf(new_val-temp_latt[j+off])-2*__cosf(old_val-temp_latt[j+off]);
				}
				delta_nrg /= latt_len;
				if(curand_uniform(&s) < exp(-delta_nrg/temp) || delta_nrg < 0) {
				latt[rand_ind] = new_val;
				}
			*/

void find_xy_parameters(int num_temps, float **latt_arr, unsigned int latt_len, unsigned int num_steps, float *Enrg, float *Magn) {
  cudaError_t cuda_ret;
  dim3 grid_dim = dim3((int)ceil(((float)latt_len)/BLOCK_SIZE), 1, 1);
  dim3 block_dim = dim3(BLOCK_SIZE, 1, 1);

  curandState *states;
  cuda_ret = cudaMalloc((void **)&states, latt_len*sizeof(curandState));
  if(cuda_ret != cudaSuccess) FATAL("Unable to alocate device memory");

  iterate_nrg<<<grid_dim, block_dim>>>(num_temps, latt_arr, latt_len, num_steps, states, Enrg);
  cudaDeviceSynchronize();
	for(int i = 0; i < num_temps; i++)
		calc_energy<<<grid_dim, block_dim>>>(latt_arr, latt_len, Enrg+i, i);
  cudaFree(states);

}


