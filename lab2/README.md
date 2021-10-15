# LAB2

## Specification
Consider the following code for "Jacobi relaxation" with input cube b and output cube a.  For each element to be computed, a function of the neighboring values (defined by a "stencil") is calculated. Assume that elements outside the input cube are 0.

```
float a[n,n,n], b[n,n,n];
for (i=0; i<n-1; i++)
    for (j=0; j<n-1; j++)
        for (k=0; k<n-1; k++) {
            a[i,j,k]=(float)0.8*(b[i-1,j,k]+b[i+1,j,k]+b[i,j-1,k] +
                  b[i,j+1,k]+b[i,j,k-1]+b[i,j,k+1]);
        }
```
This calculation should be done on the GPU.

Your program should have one argument which specifies the size n.

Your program should work on any input cube b, but for your testing, b should be initialized (CPU-side) as follows:
```
b[i][j][k] =  ((i+j+k)%10)*(float)1.1
```

Your program should output one line with two numbers on it separated by a blank:

1. The sum over all elements of the cube of `a[i][j][k] * (((i+j+k)%10)?1:-1)` using printf format "%lf".
2. The total time, in milliseconds rounded to an integer, measured from just before the first data is transferred to the GPU to the when the result matrix has been completely transferred back to the CPU.

Again, any and all optimizations are permissible...

Upload your program using the file name <student_no>.cu.