#include"support.h"

#define norm_vect_len(m) (sqrt(pow((m).x, 2) + pow((m).y, 2))/(m).num_elem)
#define BLOCK_SIZE 1024

typedef struct {
  float x;
  float y;
  int num_elem;
} magn_t;


__device__ float global_temp[BLOCK_SIZE];
__global__ void calc_energy(float *latt, unsigned int latt_len, float *nrg) {
  unsigned int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ float local_temp[BLOCK_SIZE/2];
  if(threadIdx.x < BLOCK_SIZE/2) local_temp[threadIdx.x] = 0.0;

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

__global__ void iterate_nrg(float temp, float *latt, unsigned int latt_len, int *rand_inds, float *rand_elems, float *rands, float *nrg) {
  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(index >= latt_len) return;
  int rand_ind = rand_inds[index];
  float new_val = rand_elems[index];
  float delta_nrg = 2.*__cosf(new_val-latt[rand_ind])-2.;
  for(int i = 0; i < latt_len; i++)
    delta_nrg -= 2*__cosf(new_val-latt[i])-2*__cosf(latt[rand_ind]-latt[i]);
  delta_nrg /= latt_len;
  if(rands[index] < exp(-delta_nrg/temp) || delta_nrg < 0) {
    latt[rand_ind] = new_val;
  }
}
	
void find_xy_parameters(float temp, float *latt, unsigned int latt_len, unsigned int num_steps, float *nrg, float *mag) {
  cudaError_t cuda_ret;
  dim3 grid_dim = dim3((int)ceil(((float)latt_len)/BLOCK_SIZE), 1, 1);
  dim3 block_dim = dim3(BLOCK_SIZE, 1, 1);
  
  int arr_len = latt_len;

  int *rand_inds_d;
  float *rand_elems_d, *rands_d;
  cuda_ret = cudaMalloc((void **)&rand_inds_d, arr_len * sizeof(int));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&rand_elems_d, arr_len * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&rands_d, arr_len * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

  int *rand_ind_arr = (int *)malloc(arr_len*sizeof(float));
  float *rand_spin_arr = (float *)malloc(arr_len*sizeof(float));
  float *rand_arr = (float *)malloc(arr_len*sizeof(float));

  for(int i = 0; i < num_steps; i++) {
      
    for(int j = 0; j < arr_len; j++) {
      rand_ind_arr[j] = j;//rand_latt_ind();
      rand_spin_arr[j] = rand_latt_elem();
      rand_arr[j] = uniform();
    }
    cuda_ret = cudaMemcpy(rand_inds_d, rand_ind_arr, arr_len * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy from host to device");
    cuda_ret = cudaMemcpy(rand_elems_d, rand_spin_arr, arr_len * sizeof(float), cudaMemcpyHostToDevice);    
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy from host to device");
    cuda_ret = cudaMemcpy(rands_d, rand_arr, arr_len * sizeof(float), cudaMemcpyHostToDevice);    
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy from host to device");
    cudaDeviceSynchronize();

    iterate_nrg<<<grid_dim, block_dim>>>(temp, latt, latt_len, rand_inds_d, rand_elems_d, rands_d, nrg);
  }
	cudaDeviceSynchronize();
  calc_energy<<<grid_dim, block_dim>>>(latt, latt_len, nrg);

	free(rand_ind_arr);
	free(rand_spin_arr);
	free(rand_arr);
}


