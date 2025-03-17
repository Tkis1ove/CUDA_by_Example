#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../common/book.h"

#define BLOCK_SIZE 1024
#define CEIL(a, b) ((a + b - 1)/b)

__global__ void numsAdd(int* nums, int n, int* sum){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    if(tid >= n) return;

    __shared__ int shareNums[BLOCK_SIZE];

    shareNums[tx] =  tid < n ? nums[tid] : 0;

    for(int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1){
        __syncthreads();
        if(tx < offset){
            shareNums[tx] += shareNums[tx + offset];
        }
    }

    __syncthreads();
    if(tx == 0) atomicAdd(sum, shareNums[0]);
}

int main(int argc, char** argv){

    int n = atoi(argv[1]);
    int* nums = (int*)malloc(sizeof(int) * n);
    int sum = 0;
    int d_sum = 0;

    int* device_nums, *device_sum;
    HANDLE_ERROR(cudaMalloc((void**)&device_nums, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&device_sum, sizeof(int)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i = 0; i < n; i++){
        nums[i] = rand() % 100;
    }

    HANDLE_ERROR(cudaMemcpy(device_nums, nums, sizeof(int) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(device_sum, &sum, sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(CEIL(n, BLOCK_SIZE));

    cudaEventRecord(start,0);
    float time_elapsed1=0.0;

    for(int i = 0; i < n; i++){
        sum += nums[i];
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed1,start,stop);
    printf("CPU time: %f us\n", time_elapsed1*1000);

    cudaEventRecord(start,0);
    float time_elapsed2=0.0;

    numsAdd<<<grid, block>>>(device_nums, n, device_sum);
    cudaDeviceSynchronize();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed2,start,stop);
    printf("GPU time: %f us\n", time_elapsed2*1000);

    HANDLE_ERROR(cudaMemcpy(&d_sum, device_sum, sizeof(int), cudaMemcpyDeviceToHost));

    if(sum == d_sum) printf("CPU = %d, GPU = %d, You are right!\n",sum,d_sum);
    else printf("CPU = %d, GPU = %d, You are wrong.\n",sum,d_sum);

    cudaFree(device_nums);
    cudaFree(device_sum);

    free(nums);

    return 0;
}