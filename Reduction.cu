#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CPU reduction function
void cpu_reduction(float *data, int size, float *result) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    *result = sum;
}

// GPU reduction function using shared memory
__global__ void gpu_reduction_shared(float *data, int size, float *result) {
    __shared__ float shared_data[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        shared_data[threadIdx.x] = data[idx];
    }
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

// GPU reduction function using multi-kernel
__global__ void gpu_reduction_multi(float *data, int size, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = data[idx];
    }
}

// GPU reduction function with less thread divergence
__global__ void gpu_reduction_less_divergence(float *data, int size, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += data[idx + i * size];
        }
        result[idx] = sum;
    }
}

int main() {
    int size = 1024;
    float *data = (float *)malloc(size * sizeof(float));
    float *result = (float *)malloc(sizeof(float));

    // Initialize data
    for (int i = 0; i < size; i++) {
        data[i] = i;
    }

    // CPU reduction
    cpu_reduction(data, size, result);
    printf("CPU reduction result: %f\n", *result);

    // GPU reduction using shared memory
    float *d_data, *d_result;
    cudaMalloc((void **)&d_data, size * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    gpu_reduction_shared<<<size / 256, 256>>>(d_data, size, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU reduction using shared memory result: %f\n", *result);

    // GPU reduction using multi-kernel
    cudaMalloc((void **)&d_result, size * sizeof(float));
    gpu_reduction_multi<<<size / 256, 256>>>(d_data, size, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU reduction using multi-kernel result: %f\n", *result);

    // GPU reduction with less thread divergence
    gpu_reduction_less_divergence<<<size / 256, 256>>>(d_data, size, d_result);
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU reduction with less thread divergence result: %f\n", *result);

    // Free memory
    free(data);
    free(result);
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}
