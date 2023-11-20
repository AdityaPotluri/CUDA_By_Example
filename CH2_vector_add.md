## Vector Add

The task of vector adding is relatively simple. Given a vector A and a vector B add them together and store the result in vector C.
A = [0,1,2,3] B = [4,5,6,7] and so C = [4,6,8,10]. However we will also go into how to parallelize this in CUDA and control communication
between our CPU(host) and GPU(device).
  



### Host-Device
![imagename](https://avabodha.in/content/images/2021/07/image-29.png)

We start by looking into the basic structure of a CUDA C program, how does a host(CPU) and device(GPU) communicate with each other?
Any CUDA-C program contains a mix of host-code and device code. Host-code is our traditional C program meant for execution on a CPU.
Device code on the other hand has functions called kernels that are executed in a data-parallel manner. These kernels launch a large number 
of threads which collectively make up a grid. 

![imagename](https://nyu-cds.github.io/python-gpu/fig/01-cpugpuarch.png)
The image above illustrates how a GPU takes advantage of parallelization. While a CPU may have a couple of cores, lets say around 8-12 in the 
case of the Intel Xeon Silver, an NVIDIA H200 Tensor Core GPU has 6912 cores. Using these cores we can launch a large number of threads and for 
problems that lend themselves well to data parallelization we can see a signficant improvement in performance.
  


### Device global memory and data transfer
We now return to the problem. If we have two vectors on our CPU that we would like to add together, how do we offload this task to the GPU.
How do we allocate and free memory on the device(GPU), and how do we transfer memory back and forth. We will start by writing a basic outline 
to the program.

void vecAdd(float* A, float* B, float* C, int n) {
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Part 1: Allocate device memory for A, B, and C then transferring from host to device
    

    // Part 2: Call Kernel - to launch a grid of threads on GPU
    // perform actual vector addition


    // Part 3: Copy C from the device memory
    // Free device memory
}
  


#### Part 1 + 3 Allocating and Transferring memory for A,B, and C
To do this we introduce **cudaMalloc()** and **cudaFree()**. cudaMalloc takes two arguements, the first of which is the address of a pointer 
casted to a void pointer. When cudaMalloc returns the original pointer will be modified to point to a region of device memory that it has been 
allocated to point to. The second arguement is the size in bytes of memory requested. cudaFree takes one argument, the pointer to your data.

For example:
    float *A_d;
    int size = n * sizeof(float);
    // notice that &A_d is a pointer to a pointer of the data that we have.
    // this means that the underlying location of the pointer is what is being modified
    cudaMalloc((void**) &A_d, size);
    // cudaFree() simply takes in the pointer to the data and adds it back to the available pool of memory
    cudaFree(A_d);

We still have one more problem. Now that our host code has allocated space on our device, how do we transfer the data we have on our CPU to our GPU.
  

**cudaMemcpy()**
    - memory data transfer
    - Four params
        - pointer to destination
        - pointer to source
        - Number of bytes copied
        - Type of transfer(symbolic constants predefined by CUDA programming environment)
            - cudaMemcpyHostToDevice
            - cudaMemcpyDeviceToHost
  

With this function we can fill in Part 1 and 3 of our function.


    void vecAdd(float* A, float* B, float* C, int n) {
        int size = n * sizeof(float);
        float *d_A, *d_B, *d_C;

        // Allocate device memory for A, B, and C
        // Copy A and B to device memory
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);

        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

        // Part 2: Call Kernel - to launch a grid of threads on GPU
        // perform actual vector addition


        // Part 3: Copy C from the device memory
        // Free device memory
        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
  

  
### Part 2 Launching kernel function and threading
To launch our threading functions, we have to understand how threads are organized in CUDA. When a program's host code calls 
a kernel, the CUDA runtime system launches a grid of threads, these grids are an array of thread blocks all of the same size.
The thread blocks have a max size dependent on your hardware i.e 1024. To help us handle thread blocks CUDA gives us *blockDim*
, a struct with three unsigned integer fields (x, y, and z). The dimensions of our *blockDim* generally match the dimensions of 
the data we are trying to process. For example if we want to process a 2D 100x220 image we would use just x and y set to 100 and 
200 respectively. Alternatively a 3D render of 100x250x300 would have x = 100, y = 250, z = 300. The x value represents the amount
of threads in each block, and is usually set to a multiple of 32 for hardware efficiency reasons.


![imagename] (https://ibb.co/MfDYbKx)
CUDA kernels also have access to two or more built-in variables (threadIdx and blockIdx) that allow threads to distinguish themselves 
from each other, and to determine the area of data each thread can work on. For example in the above picture the first thread in each block
has threadIdx of 0, and all the threads in the first block have a blockIdx of 0.







