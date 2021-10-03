# LAB1

## Program Specification
Write a CUDA program that does the following calculations on two  input matrices, X and Y, to generate an output matrix, Z:

Z[i,j] = X[i-1,j] + X[i,j] + X[i+1,j] – Y[i,j-2] – Y[i,j-1] – Y[i,j]

where out of bound elements should be assumed to be 0; that is:

X[x,y] = 0 and Y[x,y] = 0   if x<0, y<0, x>=n, or y>=m

All three matrices of type float.

 

Your CUDA program should accept two arguments:

An integer specifying the number of rows in the matrices
An integer specifying the number of columns in the matrices
Each time the CUDA program is invoked, it shall invoke a GPU kernel called f_siggen(), exactly once to perform the above calculation.

 

Your CUDA program should output 5 numbers on one line terminated with the newline character and then exit:

<total_GPU_time> <CPU_GPU_transfer_time> <kernel_time> <GPU_CPU_transfer_time> <Z-value> <nl>

Each of these numbers are defined as follows (and the code provided below shows how to obtain these numbers):

<total_GPU_time>: the time in seconds measured from just before the first data is transferred to the GPU to right after the last data has been transferred back to the CPU.

<CPU_GPU_transfer_time>: the time in seconds it takes to transfer the two input matrices to the GPU.

<kernel_time>: the time in seconds it takes the GPU to execute the kernel.
<GPU_CPU_transfer_time>: the time in seconds it takes the GPU to transfer the result matrix back to the CPU.
<Z-value>: the value of Z[5,5].
The numbers should be output with 6 digits of precision (“%.6f”). All four numbers must be output only one line and be separated by one space.

If an error occurs, then your program should output one line starting with “Error: ” followed by a description of the error before exiting.

 

The two input matrices shall be initialized host-side (before copying them to GPU global memory) as follows:
```
h_X[i,j] = (float) (i+j)/2.0 ;
h_Y[i,j] = (float) 3.25*(i+j) ;
```
The result matrix produced on the GPU, here called d_Z, should be copied back to CPU memory (perhaps into h_dZ) and -- to check correctness -- should be compared against the result of the same calculation performed entirely CPU-side (perhaps generating h_hZ). Your code must do this comparison, and if there is a discrepancy, an error message must be output (instead of the numbers).

## Deliverables
Your entire program must be contained within one file and submitted as <student-number>.cu by uploading this file. It should #include only standard include files.

 

## Some Hints
Please ensure that your program can correctly handle matrices of the following sizes: 16,384 x 16,384 (a nice square matrix), 32,768 x 8,192, and 30 x 8,947,850 . Your program should take significantly less than 60 seconds to execute --- when we test your code, we will cut off your execution after 60 seconds.

 

The following function may be useful to collect timing information. It is used CPU-side only.
```
// time stamp function in seconds
#include <sys/time.h>
double getTimeStamp() {
    struct timeval  tv ; gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
```

Remember that this assignment is competitive. The faster your program runs, the higher the mark you will receive. Hence, some things you may want to optimize:

optimal block size and shape
coalesced global memory accesses
thread divergence
number of d_Z elements calculated per thread