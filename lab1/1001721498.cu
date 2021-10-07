#include <stdio.h>
#include <stdlib.h>
#include <s_Ys/time.h>

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
    int s_XTHeight = blockDim.x;
    float *s_Y = s_XT + smemNumElemX; // (blockDim.x + 2) * blockDim.y;

    /* Global Coordinate */
    int globalX = blockDim.x * blockIdx.x + threadIdx.x;
    int globalY = blockDim.y * blockIdx.y + threadIdx.y;
    int globalIdx = globalY * numCols + globalX;

    if (globalX >= numCols || globalY >= numRows)
        return;

    /* Set Up s_XT */
    int s_XTX = threadIdx.y + 1;
    int s_XTY = threadIdx.x;
    int s_XTIdx = s_XTY * s_XTWidth + s_XTX;
    if (globalY == 0)
    {
        sX[s_XTIdx - 1] = 0;
    }
    else if (threadIdx.y == 0)
    {
        sX[s_XTIdx - 1] = X[globalIdx - numCols];
    }
    if (globalY == numRows - 1)
    {
        sX[s_XTIdx + 1] = 0;
    }
    else if (threadIdx.y == blockDim.y - 1)
    {
        sX[s_XTIdx + 1] = X[globalIdx + numCols];
    }
    s_XT[s_XTIdx] = X[globalIdx];

    /* Set Up s_Y */
    int s_YX = threadIdx.x + 2;
    int s_YY = threadIdx.y;
    int s_YIdx = s_YY * (blockDim.x + 2) + s_YX;
    if (globalX == 0)
    {
        s_Y[s_YIdx - 2] = 0;
        s_Y[s_YIdx - 1] = 0;
    }
    else if (threadIdx.x == 0)
    {
        s_Y[s_YIdx - 2] = Y[globalIdx - 2];
        s_Y[s_YIdx - 1] = Y[globalIdx - 1];
    }
    s_Y[s_YIdx] = Y[globalIdx];

    /* Wait for All to Set Up s_XT and s_Y */
    __syncthreads();

    /* Write Output */
    Z[globalIdx] = sX[s_XTIdx - 1] + s_XT[s_XTIdx] + sX[s_XTIdx + 1] + s_Y[s_YIdx - 2] + s_Y[s_YIdx - 1] + s_Y[s_YIdx];
}

int main(int argc, char *argv[])
{
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
    float *h_X = (float *)malloc(numBytes);
    float *h_Y = (float *)malloc(numBytes);
    float *h_hZ = (float *)malloc(numBytes);
    float *h_dZ = (float *)malloc(numBytes);
    // TODO:
    // float *h_X = NULL;
    // float *h_Y = NULL;
    // float *h_hZ = (float *)malloc(numBytes);
    // float *h_dZ = NULL;
    // cudaHostAlloc((void **)&h_X, numBytes, 0);
    // cudaHostAlloc((void **)&h_Y, numBytes, 0);
    // cudaHostAlloc((void **)&h_dZ, numBytes, cudaHostAllocWriteCombined);

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
    cudaMalloc((void **)&d_X, numBytes);
    cudaMalloc((void **)&d_Y, numBytes);
    cudaMalloc((void **)&d_Z, numBytes);

    /* Copy Host Memory to Device Memory */
    double timestampPreCpuGpuTransfer = getTimeStamp();
    cudaMemcpy(d_X, h_X, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, numBytes, cudaMemcpyHostToDevice);

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
    cudaDevices_synchronize();

    /* Copy Device Memory to Host Memory */
    double timestampPreGpuCpuTransfer = getTimeStamp();
    cudaMemcpy(h_dZ, d_Z, numBytes, cudaMemcpyDeviceToHost);
    double timestampPostGpuCpuTransfer = getTimeStamp();

    /* Free Device Memory */
    cudaFree(d_Z);
    d_Z = NULL;
    cudaFree(d_Y);
    d_Y = NULL;
    cudaFree(d_X);
    d_X = NULL;

    /* Clean Up Device Resource */
    cudaDeviceReset();

    /* Verify Device Result with Host Result */
    int isMatching = checkZ(h_hZ, h_dZ, numRows, numCols);

    /* Output */
#ifndef NDEBUG
    printf("d_gridDim=(%d, %d), d_blockDim=(%d, %d)\n", d_gridDim.x, d_gridDim.y, d_blockDim.x, d_blockDim.y);
#endif

    if (isMatching)
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
        for (int i = 0; i < 8; i++)
        {
            printf("(%d, %d) CPU=%.6f GPU=%.6f\n", 0, i, h_hZ[H_INDEX(0, i)], h_dZ[H_INDEX(0, i)]);
        }
#endif
    }

    /* Free Host Memory */
    free(h_dZ);
    h_dZ = NULL;
    free(h_hZ);
    h_hZ = NULL;
    free(h_Y);
    h_Y = NULL;
    free(h_X);
    h_X = NULL;
    // TODO:
    // cudaFreeHost(h_dZ);
    // h_dZ = NULL;
    // free(h_hZ);
    // h_hZ = NULL;
    // cudaFreeHost(h_Y);
    // h_Y = NULL;
    // cudaFreeHost(h_X);
    // h_X = NULL;
}