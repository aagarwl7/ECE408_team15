#include"support.h"

#define norm_vect_len(m) (sqrt(pow((m).x, 2) + pow((m).y, 2))/(m).num_elem)
#define BLOCK_SIZE 1024

typedef struct {
  float x;
  float y;
  int num_elem;
} magn_t;

__global__ void calc_energy(float *latt, unsigned int latt_len, float *nrg) {
  unsigned int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  float retval = 2.0;
  float spin = latt[index];
  for(int i = 0; i < latt_len; i++)
    retval -= 2*cos(latt[i] - spin);
  retval /= latt_len;
  atomicAdd(nrg, retval);
}

__global__ void iterate_nrg(float temp, float *latt, unsigned int latt_len, int *rand_inds, float *rand_elems, float *rands, float *nrg) {
  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rand_ind = rand_inds[index];
  float new_val = rand_elems[index];
  float delta_nrg = 2.*cos(new_val-latt[rand_ind])-2.;
  for(int i = 0; i < latt_len; i++)
    delta_nrg -= 2*cos(new_val-latt[i])-2*cos(latt[rand_ind]-latt[i]);
  delta_nrg /= latt_len;
  if(rands[index] < exp(-delta_nrg/temp) || delta_nrg < 0) {
    latt[rand_ind] = new_val;
  }
}
	
void find_xy_parameters(float temp, float *latt, unsigned int latt_len, unsigned int num_iter, float *nrg, float *mag) {
  cudaError_t cuda_ret;
  dim3 grid_dim = dim3((int)ceil(((float)latt_len)/BLOCK_SIZE), 1, 1);
  dim3 block_dim = dim3(BLOCK_SIZE, 1, 1);

  int arr_len = grid_dim.x*block_dim.x;

  int *rand_inds_d;
  float *rand_elems_d, *rands_d;
  cuda_ret = cudaMalloc((void **)&rand_inds_d, arr_len * sizeof(int));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&rand_elems_d, arr_len * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  cuda_ret = cudaMalloc((void **)&rands_d, arr_len * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

  int rand_ind_arr[arr_len];
  float rand_spin_arr[arr_len];
  float rand_arr[arr_len];

  calc_energy<<<grid_dim, block_dim>>>(latt, latt_len, nrg);
  for(int i = 0; i < num_iter/(grid_dim.x*block_dim.x); i++) {
    
    for(int i = 0; i < arr_len; i++) {
      rand_ind_arr[i] = rand_latt_ind();
      rand_spin_arr[i] = rand_latt_elem();
      rand_arr[i] = uniform();
    }
    cuda_ret = cudaMemcpy(rand_inds_d, rand_ind_arr, arr_len * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy from host to device");
    cuda_ret = cudaMemcpy(rand_elems_d, rand_spin_arr, arr_len * sizeof(float), cudaMemcpyHostToDevice);    
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy from host to device");
    cuda_ret = cudaMemcpy(rands_d, rand_arr, arr_len * sizeof(float), cudaMemcpyHostToDevice);    
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy from host to device");
    cudaDeviceSynchronize();

    iterate_nrg<<<grid_dim, block_dim>>>(temp, latt, latt_len, rand_inds_d, rand_elems_d, rands_d, nrg);
    calc_energy<<<grid_dim, block_dim>>>(latt, latt_len, nrg);
  }
}
