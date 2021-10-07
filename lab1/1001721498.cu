#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Comment out this line to enable debug mode
// #define NDEBUG

/* time stamp function in seconds */
__host__ double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

__host__ void initX(float *X, int numRows, int numCols)
{
    int lastIBase = (numRows + 1) * numCols;
    for (int j = 0; j < numCols; j++)
    {
        X[j] = 0;
        X[lastIBase + j] = 0;
    }
    for (int i = 0; i < numRows; i++)
    {
        int iBase = (i + 1) * numCols;
        for (int j = 0; j < numCols; j++)
        {
            // h_X_old[i,j] = (float) (i+j)/2.0;
            X[iBase + j] = (float)(i + j) / 2.0;
        }
    }
}

__host__ void initY(float *Y, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        int iBase = i * (numCols + 2);
        Y[iBase] = 0;
        Y[iBase + 1] = 0;
        for (int j = 0; j < numCols; j++)
        {
            // h_Y_old[i,j] = (float) 3.25*(i+j);
            Y[iBase + j + 2] = (float)3.25 * (i + j);
        }
    }
}

#define H_ADJ_INDEX_X(i, j) ((i) + 1) * numCols + (j)
#define H_ADJ_INDEX_Y(i, j) (i) * (numCols + 2) + (j) + 2
#define H_INDEX(i, j) (i) * numCols + (j)

__host__ void f_siggen_reference(float *X, float *Y, float *Z, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            // Z[i,j] = X[i-1,j] + X[i,j] + X[i+1,j] – Y[i,j-2] – Y[i,j-1] – Y[i,j]
            Z[H_INDEX(i, j)] =
                X[H_ADJ_INDEX_X(i - 1, j)] +
                X[H_ADJ_INDEX_X(i, j)] +
                X[H_ADJ_INDEX_X(i + 1, j)] -
                Y[H_ADJ_INDEX_Y(i, j - 2)] -
                Y[H_ADJ_INDEX_Y(i, j - 1)] -
                Y[H_ADJ_INDEX_Y(i, j)];
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
                printf("(i=%d, j=%d) Expected=%f Actual=%f\n", i, j, E[ibase + j], A[ibase + j]);
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
    int globalXIdx = (globalY + 1) * numCols + globalX;
    int globalYIdx = globalY * (numCols + 2) + globalX + 2;

    if (globalX >= numCols || globalY >= numRows)
        return;

    /* Set Up s_XT */
    int s_XTx = threadIdx.y + 1;
    int s_XTy = threadIdx.x;
    int s_XTIdx = s_XTy * s_XTWidth + s_XTx;
    if (threadIdx.y == 0)
    {
        s_XT[s_XTIdx - 1] = X[globalXIdx - numCols];
    }
    if (threadIdx.y == blockDim.y - 1 || globalY == numRows - 1)
    {
        s_XT[s_XTIdx + 1] = X[globalXIdx + numCols];
    }
    s_XT[s_XTIdx] = X[globalXIdx];

    /* Set Up s_Y */
    int s_Yx = threadIdx.x + 2;
    int s_Yy = threadIdx.y;
    int s_YIdx = s_Yy * (blockDim.x + 2) + s_Yx;
    if (threadIdx.x == 0)
    {
        s_Y[s_YIdx - 2] = Y[globalYIdx - 2];
        s_Y[s_YIdx - 1] = Y[globalYIdx - 1];
    }
    s_Y[s_YIdx] = Y[globalYIdx];

    /* Wait for All to Set Up s_XT and s_Y */
    __syncthreads();

    /* Write Output */
    Z[globalIdx] = s_XT[s_XTIdx - 1] + s_XT[s_XTIdx] + s_XT[s_XTIdx + 1] - s_Y[s_YIdx - 2] - s_Y[s_YIdx - 1] - s_Y[s_YIdx];
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

    int numRowsX = numRows + 2;
    int numColsX = numCols;
    size_t numElemX = numRowsX * numColsX;
    size_t numBytesX = numElemX * sizeof(float);

    int numRowsY = numRows;
    int numColsY = numCols + 2;
    size_t numElemY = numRowsY * numColsY;
    size_t numBytesY = numElemY * sizeof(float);

#ifndef NDEBUG
    printf("numRows=%d, numCols=%d, numElem=%ld, numBytes=%ld\n", numRows, numCols, numElem, numBytes);
    printf("numRowsX=%d, numColsX=%d, numElemX=%ld, numBytesX=%ld\n", numRowsX, numColsX, numElemX, numBytesX);
    printf("numRowsY=%d, numColsY=%d, numElemY=%ld, numBytesY=%ld\n", numRowsY, numColsY, numElemY, numBytesY);
#endif

    /* Allocate Host Memory */
    float *h_X = NULL;
    float *h_Y = NULL;
    float *h_hZ = (float *)malloc(numBytes);
    float *h_dZ = NULL;
    error = error || cudaHostAlloc((void **)&h_X, numBytesX, 0);
    error = error || cudaHostAlloc((void **)&h_Y, numBytesY, 0);
    error = error || cudaHostAlloc((void **)&h_dZ, numBytes, 0);
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
    error = error || cudaMemcpy(d_X, h_X, numBytesX, cudaMemcpyHostToDevice);
    error = error || cudaMemcpy(d_Y, h_Y, numBytesY, cudaMemcpyHostToDevice);
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
    d_gridDim.x = (numCols - 1) / d_blockDim.x + 1;
    d_gridDim.y = (numRows - 1) / d_blockDim.y + 1;
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
    printf("d_gridDim=(%d, %d), d_blockDim=(%d, %d), d_smemNumBytes=%ld\n", d_gridDim.x, d_gridDim.y, d_blockDim.x, d_blockDim.y, d_smemNumBytes);
#endif

    if (!error)
    {
        // #ifndef NDEBUG
        //         printf("<total_GPU_time> <CPU_GPU_transfer_time> <kernel_time> <GPU_CPU_transfer_time> <Z-value> <nl>\n");
        // #endif
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
                printf("(i=%d, j=%d), CPU=%.6f, GPU=%.6f, X=%.6f, Y=%.6f\n", i, j, h_hZ[H_INDEX(i, j)], h_dZ[H_INDEX(i, j)], h_X[H_ADJ_INDEX_X(i, j)], h_Y[H_ADJ_INDEX_Y(i, j)]);
            }
        }
#endif
    }

    /* Free Host Memory */
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