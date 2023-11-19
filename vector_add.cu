#include <cuda_runtime.h>
// function just adds vectors A and B and stores result in C
// [0, 1, 2, 3] + [4, 5, 6, 7] = [4, 6, 8, 10]
void vecAdd(float* A, float* B, float* C, int n) {
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory for A, B, and C
    // Copy A and B to device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Part 2: Call Kernel - to launch a grid of threads on GPU
    // perform actual vector addition


    // Part 3: Copy C from the device memory
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}
int main() {

}