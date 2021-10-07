#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Comment out this line to enable debug mode
// #define NDEBUG

#define H_INDEX(i, j) (i) * numCols + (j)

/* time stamp function in seconds */
__host__ double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

__host__ void initX(float *X, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        int ibase = i * numCols;
        for (int j = 0; j < numCols; j++)
        {
            // h_X[i,j] = (float) (i+j)/2.0;
            X[ibase + j] = (float)(i + j) / 2.0;
        }
    }
}

__host__ void initY(float *Y, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        int ibase = i * numCols;
        for (int j = 0; j < numCols; j++)
        {
            // h_Y[i,j] = (float) 3.25*(i+j);
            Y[ibase + j] = (float)3.25 * (i + j);
        }
    }
}

__host__ float f_siggen_reference_get(float *M, int i, int j, int numRows, int numCols)
{
    if (i < 0 || i >= numRows || j < 0 || j >= numCols)
    {
        return 0;
    }
    return M[i * numCols + j];
}

__host__ void f_siggen_reference(float *X, float *Y, float *Z, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        int ibase = i * numCols;
        for (int j = 0; j < numCols; j++)
        {
            // Z[i,j] = X[i-1,j] + X[i,j] + X[i+1,j] – Y[i,j-2] – Y[i,j-1] – Y[i,j]
            Z[ibase + j] =
                f_siggen_reference_get(X, i - 1, j, numRows, numCols) +
                f_siggen_reference_get(X, i, j, numRows, numCols) +
                f_siggen_reference_get(X, i + 1, j, numRows, numCols) -
                f_siggen_reference_get(Y, i, j - 2, numRows, numCols) -
                f_siggen_reference_get(Y, i, j - 1, numRows, numCols) -
                f_siggen_reference_get(Y, i, j, numRows, numCols);
        }
    }
}

__host__ int checkZ(float *E, float *A, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        int ibase = i * numCols;
        for (int j = 0; j < numCols; j++)
        {
            if (E[ibase + j] != A[ibase + j])
            {
                return 0;
            }
        }
    }
    return 1;
}

__global__ void f_siggen(float *X, float *Y, float *Z, int numRows, int numCols, int smemNumElemX)
{
    extern __shared__ float s_data[];
    float *s_XT = s_data; // blockDim.x * (blockDim.y + 2);
    int s_XTWidth = (blockDim.y + 2);
    // int s_XTHeight = blockDim.x;
    float *s_Y = s_XT + smemNumElemX; // (blockDim.x + 2) * blockDim.y;

    /* Global Coordinate */
    int globalX = blockDim.x * blockIdx.x + threadIdx.x;
    int globalY = blockDim.y * blockIdx.y + threadIdx.y;
    int globalIdx = globalY * numCols + globalX;

    if (globalX >= numCols || globalY >= numRows)
        return;

    /* Set Up s_XT */
    int s_XT_x = threadIdx.y + 1;
    int s_XT_y = threadIdx.x;
    int s_XT_idx = s_XT_y * s_XTWidth + s_XT_x;
    if (globalY == 0)
    {
        s_XT[s_XT_idx - 1] = 0;
    }
    else if (threadIdx.y == 0)
    {
        s_XT[s_XT_idx - 1] = X[globalIdx - numCols];
    }
    if (globalY == numRows - 1)
    {
        s_XT[s_XT_idx + 1] = 0;
    }
    else if (threadIdx.y == blockDim.y - 1)
    {
        s_XT[s_XT_idx + 1] = X[globalIdx + numCols];
    }
    s_XT[s_XT_idx] = X[globalIdx];

    /* Set Up s_Y */
    int s_Y_x = threadIdx.x + 2;
    int s_Y_y = threadIdx.y;
    int s_Y_idx = s_Y_y * (blockDim.x + 2) + s_Y_x;
    if (globalX == 0)
    {
        s_Y[s_Y_idx - 2] = 0;
        s_Y[s_Y_idx - 1] = 0;
    }
    else if (threadIdx.x == 0)
    {
        s_Y[s_Y_idx - 2] = Y[globalIdx - 2];
        s_Y[s_Y_idx - 1] = Y[globalIdx - 1];
    }
    s_Y[s_Y_idx] = Y[globalIdx];

    /* Wait for All to Set Up s_XT and s_Y */
    __syncthreads();

    /* Write Output */
    Z[globalIdx] = s_XT[s_XT_idx - 1] + s_XT[s_XT_idx] + s_XT[s_XT_idx + 1] - s_Y[s_Y_idx - 2] - s_Y[s_Y_idx - 1] - s_Y[s_Y_idx];
}

int main(int argc, char *argv[])
{
    int error = 0;
    /* Get Dimension */
    if (argc != 3)
    {
        printf("Error: The number of arguments is not exactly 2\n");
        return 0;
    }
    int numRows = atoi(argv[1]);
    int numCols = atoi(argv[2]);
    size_t numElem = numRows * numCols;
    size_t numBytes = numElem * sizeof(float);

#ifndef NDEBUG
    printf("numRows=%d, numCols=%d, numElem=%ld, numBytes=%ld\n", numRows, numCols, numElem, numBytes);
#endif

    /* Allocate Host Memory */
    // float *h_X = (float *)malloc(numBytes);
    // float *h_Y = (float *)malloc(numBytes);
    // float *h_hZ = (float *)malloc(numBytes);
    // float *h_dZ = (float *)malloc(numBytes);
    float *h_X = NULL;
    float *h_Y = NULL;
    float *h_hZ = (float *)malloc(numBytes);
    float *h_dZ = NULL;
    error = error || cudaHostAlloc((void **)&h_X, numBytes, 0);
    error = error || cudaHostAlloc((void **)&h_Y, numBytes, 0);
    error = error || cudaHostAlloc((void **)&h_dZ, numBytes, cudaHostAllocWriteCombined);
    if (error)
    {
        printf("Error: cudaHostAlloc returns error\n");
        return 0;
    }

    /* Initialize Host Memory */
    initX(h_X, numRows, numCols);
    initY(h_Y, numRows, numCols);
#ifndef NDEBUG
    double timestampPreCpuKernel = getTimeStamp();
#endif
    f_siggen_reference(h_X, h_Y, h_hZ, numRows, numCols);
#ifndef NDEBUG
    double timestampPostCpuKernel = getTimeStamp();
    printf("CPU=%.6fsec\n", timestampPostCpuKernel - timestampPreCpuKernel);
#endif

    /* Allocate Device Memory */
    float *d_X = NULL;
    float *d_Y = NULL;
    float *d_Z = NULL;
    error = error || cudaMalloc((void **)&d_X, numBytes);
    error = error || cudaMalloc((void **)&d_Y, numBytes);
    error = error || cudaMalloc((void **)&d_Z, numBytes);
    if (error)
    {
        printf("Error: cudaMalloc returns error\n");
        return 0;
    }

    /* Copy Host Memory to Device Memory */
    double timestampPreCpuGpuTransfer = getTimeStamp();
    error = error || cudaMemcpy(d_X, h_X, numBytes, cudaMemcpyHostToDevice);
    error = error || cudaMemcpy(d_Y, h_Y, numBytes, cudaMemcpyHostToDevice);
    if (error)
    {
        printf("Error: cudaMemcpy returns error\n");
        return 0;
    }

    /* Run Kernel */
    double timestampPreKernel = getTimeStamp();
    dim3 d_blockDim;
    d_blockDim.x = 32;
    d_blockDim.y = 32;
    dim3 d_gridDim;
    d_gridDim.x = (numCols + 1) / d_blockDim.x;
    d_gridDim.y = (numRows + 1) / d_blockDim.y;
    int d_smemNumElemX = d_blockDim.x * (d_blockDim.y + 2);
    int d_smemNumElemY = (d_blockDim.x + 2) * d_blockDim.y;
    size_t d_smemNumBytes = (d_smemNumElemX + d_smemNumElemY) * sizeof(float);
    f_siggen<<<d_gridDim, d_blockDim, d_smemNumBytes>>>(d_X, d_Y, d_Z, numRows, numCols, d_smemNumElemX);
    cudaDeviceSynchronize();

    /* Copy Device Memory to Host Memory */
    double timestampPreGpuCpuTransfer = getTimeStamp();
    error = error || cudaMemcpy(h_dZ, d_Z, numBytes, cudaMemcpyDeviceToHost);
    if (error)
    {
        printf("Error: cudaMemcpy returns error\n");
        return 0;
    }
    double timestampPostGpuCpuTransfer = getTimeStamp();

    /* Free Device Memory */
    cudaFree(d_Z);
    d_Z = NULL;
    cudaFree(d_Y);
    d_Y = NULL;
    cudaFree(d_X);
    d_X = NULL;

    /* Verify Device Result with Host Result */
    error = error || !checkZ(h_hZ, h_dZ, numRows, numCols);

    /* Output */
#ifndef NDEBUG
    printf("d_gridDim=(%d, %d), d_blockDim=(%d, %d)\n", d_gridDim.x, d_gridDim.y, d_blockDim.x, d_blockDim.y);
#endif

    if (!error)
    {
#ifndef NDEBUG
        printf("<total_GPU_time> <CPU_GPU_transfer_time> <kernel_time> <GPU_CPU_transfer_time> <Z-value> <nl>\n");
#endif
        float totalGpuElapased = timestampPostGpuCpuTransfer - timestampPreCpuGpuTransfer;
        float cpuGpuTransferElapsed = timestampPreKernel - timestampPreCpuGpuTransfer;
        float kernelElapsed = timestampPreGpuCpuTransfer - timestampPreKernel;
        float gpuCpuTransferElapsed = timestampPostGpuCpuTransfer - timestampPreGpuCpuTransfer;
        int zValueI = 5;
        int zValueJ = 5;
        float zValue = h_dZ[H_INDEX(zValueI, zValueJ)];
        printf("%.6f %.6f %.6f %.6f %.6f\n", totalGpuElapased, cpuGpuTransferElapsed, kernelElapsed, gpuCpuTransferElapsed, zValue);
    }
    else
    {
        printf("Error: GPU result does not with CPU result\n");
#ifndef NDEBUG
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                printf("(i=%d, j=%d), CPU=%.6f, GPU=%.6f, X=%.6f, Y=%.6f\n", i, j, h_hZ[H_INDEX(i, j)], h_dZ[H_INDEX(i, j)], h_X[H_INDEX(i, j)], h_Y[H_INDEX(i, j)]);
            }
        }
#endif
    }

    /* Free Host Memory */
    // free(h_dZ);
    // h_dZ = NULL;
    // free(h_hZ);
    // h_hZ = NULL;
    // free(h_Y);
    // h_Y = NULL;
    // free(h_X);
    // h_X = NULL;
    cudaFreeHost(h_dZ);
    h_dZ = NULL;
    free(h_hZ);
    h_hZ = NULL;
    cudaFreeHost(h_Y);
    h_Y = NULL;
    cudaFreeHost(h_X);
    h_X = NULL;

    /* Clean Up Device Resource */
    cudaDeviceReset();
}