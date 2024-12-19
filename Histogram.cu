#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CPU histogram function
void cpu_histogram(float *data, int size, int *histogram) {
    for (int i = 0; i < size; i++) {
        int bin = (int)data[i];
        histogram[bin]++;
    }
}

// GPU histogram function without atomics
__global__ void gpu_histogram_no_atomics(float *data, int size, int *histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bin = (int)data[idx];
        histogram[bin]++;
    }
}

// GPU histogram function with atomics
__global__ void gpu_histogram_atomics(float *data, int size, int *histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bin = (int)data[idx];
        atomicAdd(&histogram[bin], 1);
    }
}

// GPU histogram function with strided access
__global__ void gpu_histogram_strided(float *data, int size, int *histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bin = (int)data[idx];
        atomicAdd(&histogram[bin], 1);
    }
}

int main() {
    int size = 1024;
    float *data = (float *)malloc(size * sizeof(float));
    int *histogram = (int *)malloc(256 * sizeof(int));

    // Initialize data
    for (int i = 0; i < size; i++) {
        data[i] = i % 256;
    }

    // CPU histogram
    cpu_histogram(data, size, histogram);
    printf("CPU histogram result:\n");
    for (int i = 0; i < 256; i++) {
        printf("%d ", histogram[i]);
    }
    printf("\n");

    // GPU histogram without atomics
    float *d_data;
    int *d_histogram;
    cudaMalloc((void **)&d_data, size * sizeof(float));
    cudaMalloc((void **)&d_histogram, 256 * sizeof(int));
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    gpu_histogram_no_atomics<<<size / 256, 256>>>(d_data, size, d_histogram);
    cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU histogram without atomics result:\n");
    for (int i = 0; i < 256; i++) {
        printf("%d ", histogram[i]);
    }
    printf("\n");

    // GPU histogram with atomics
    gpu_histogram_atomics<<<size / 256, 256>>>(d_data, size, d_histogram);
    cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU histogram with atomics result:\n");
    for (int i = 0; i < 256; i++) {
        printf("%d ", histogram[i]);
    }
    printf("\n");

    // GPU histogram with strided access
    gpu_histogram_strided<<<size / 256, 256>>>(d_data, size, d_histogram);
    cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU histogram with strided access result:\n");
    for (int i = 0; i < 256; i++) {
        printf("%d ", histogram[i]);
    }
    printf("\n");

    // Free memory
    free(data);
    free(histogram);
    cudaFree(d_data);
    cudaFree(d_histogram);

    return 0;
}
