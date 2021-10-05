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

__global__ void f_siggen()
{
    // wip
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
    int numElem = numRows * numCols;

#ifndef NDEBUG
    printf("numRows=%d, numCols=%d, numElem=%d", numRows, numCols, numElem);
#endif

    /* Allocate Host Memory */
    float *h_X = (float *)malloc(sizeof(float) * numElem);
    float *h_Y = (float *)malloc(sizeof(float) * numElem);
    float *h_hZ = (float *)malloc(sizeof(float) * numElem);
    float *h_dZ = (float *)malloc(sizeof(float) * numElem);

    /* Initialize Host Memory */
    initX(h_X, numRows, numCols);
    initY(h_Y, numRows, numCols);
    f_siggen_reference(h_X, h_Y, h_hZ, numRows, numCols);

    /* Allocate Device Memory */

    /* Copy Host Memory to Device Memory */
    double timestampPreCpuGpuTransfer = getTimeStamp();

    /* Launch Kernel */
    double timestampPreKernel = getTimeStamp();
    dim3 gridDim;
    dim3 blockDim;
    size_t d_size_smem = 0;
    f_siggen<<<gridDim, blockDim, d_size_smem>>>();
    // sync

    /* Copy Device Memory to Host Memory */
    double timestampPreGpuCpuTransfer = getTimeStamp();

    double timestampPostGpuCpuTransfer = getTimeStamp();

    /* Clean Up Device Resource */

    /* Verify Device Result with Host Result */
    int isMatching = checkZ(h_hZ, h_dZ, numRows, numCols);

    /* Output */
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
        float zValue = h_dZ[zValueI * numCols + zValueJ];
        printf("%.6f %.6f %.6f %.6f %.6f\n", totalGpuElapased, cpuGpuTransferElapsed, kernelElapsed, gpuCpuTransferElapsed, zValue);
    }
    else
    {
        printf("Error: GPU result does not with CPU result\n");
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
}