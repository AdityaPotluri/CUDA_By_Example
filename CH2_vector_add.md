## Vector Add

The task of vector adding is relatively simple. Given a vector A and a vector B add them together and store the result in vector C.
A = [0,1,2,3] B = [4,5,6,7] and so C = [4,6,8,10]. However we will also go into how to parallelize this in CUDA and control communication
between our CPU(host) and GPU(device).


### Host-Device
![imagename](https://avabodha.in/content/images/2021/07/image-29.png)

We start by looking into the basic structure of a CUDA C program, how does a host(CPU) and device(GPU) communicate with each other?
Any CUDA-C program contains a mix of host-code and device code. Host-code is our traditional C program meant for execution on a CPU.
Device code on the other hand has functions called kernels that are executed in a data-parallel manner. 

![imagename]