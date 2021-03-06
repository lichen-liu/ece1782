#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Comment out this line to enable debug mode
#define NDEBUG

/* time stamp function in milliseconds */
__host__ double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

__host__ void initB(float *B, int nB)
{
    for (int i = 0; i < nB; i++)
    {
        int iIndex = i * nB * nB;
        for (int j = 0; j < nB; j++)
        {
            int ijIndex = iIndex + j * nB;
            for (int k = 0; k < nB; k++)
            {
                int ijkIndex = ijIndex + k;
                if (i == 0 || j == 0 || k == 0)
                {
                    B[ijkIndex] = 0;
                }
                else
                {
                    B[ijkIndex] = ((i - 1 + j - 1 + k - 1) % 10) * (float)1.1;
                }
            }
        }
    }
}

#define h_getB(B, nB, i, j, k) B[((i) + 1) * nB * nB + ((j) + 1) * nB + ((k) + 1)]

__host__ void jacobiRelaxationReference(float *A, float *B, int n)
{
    int nB = n + 1;
    for (int i = 0; i < n; i++)
    {
        int iIndex = i * n * n;
        for (int j = 0; j < n; j++)
        {
            int ijIndex = iIndex + j * n;
            for (int k = 0; k < n; k++)
            {
                int ijkIndex = ijIndex + k;
                if (i >= n - 1 || j >= n - 1 || k >= n - 1)
                {
                    A[ijkIndex] = 0.0;
                }
                else
                {
                    A[ijkIndex] = (float)0.8 * (h_getB(B, nB, i - 1, j, k) +
                                                h_getB(B, nB, i + 1, j, k) +
                                                h_getB(B, nB, i, j - 1, k) +
                                                h_getB(B, nB, i, j + 1, k) +
                                                h_getB(B, nB, i, j, k - 1) +
                                                h_getB(B, nB, i, j, k + 1));
                }
            }
        }
    }
}

__host__ int checkA(float *Expected, float *Actual, int n)
{
    for (int i = 0; i < n; i++)
    {
        int iIndex = i * n * n;
        for (int j = 0; j < n; j++)
        {
            int ijIndex = iIndex + j * n;
            for (int k = 0; k < n; k++)
            {
                int ijkIndex = ijIndex + k;
                if (Expected[ijkIndex] != Actual[ijkIndex])
                {
#ifndef NDEBUG
                    printf("(i=%d, j=%d, k=%d) Expected=%f Actual=%f\n", i, j, k, Expected[ijkIndex], Actual[ijkIndex]);
#endif
                    return 0;
                }
            }
        }
    }
    return 1;
}

__host__ double sumA(float *A, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        int iIndex = i * n * n;
        for (int j = 0; j < n; j++)
        {
            int ijIndex = iIndex + j * n;
            for (int k = 0; k < n; k++)
            {
                int ijkIndex = ijIndex + k;
                sum += A[ijkIndex] * (((i + j + k) % 10) ? 1 : -1);
            }
        }
    }
    return sum;
}

__global__ void jacobiRelaxation(float *A, float *B, int n, int startingI)
{
    extern __shared__ float s_data[];

    /* Global Index */
    int globalK = blockDim.x * blockIdx.x + threadIdx.x;
    int globalJ = blockDim.y * blockIdx.y + threadIdx.y;
    int globalI = blockDim.z * blockIdx.z + threadIdx.z + startingI;
    int globalIdx = globalI * n * n + globalJ * n + globalK;

    int nB = n + 1;
    int sizePerGlobalBI = nB * nB;
    int sizePerGlobalBJ = nB;
    int globalBIIndex = (globalI + 1) * sizePerGlobalBI;
    int globalBIJIndex = globalBIIndex + (globalJ + 1) * sizePerGlobalBJ;
    int globalBIdx = globalBIJIndex + (globalK + 1);

    if (globalK >= n || globalJ >= n || globalI >= n)
    {
        return;
    }

    /* Local Index */
    // int sizeI = blockDim.z + 2;
    int sizeJ = blockDim.y + 2;
    int sizeK = blockDim.x + 2;
    int sizePerLocalI = sizeJ * sizeK;
    int sizePerLocalJ = sizeK;
    int localIIndex = (threadIdx.z + 1) * sizePerLocalI;
    int localIJIndex = localIIndex + (threadIdx.y + 1) * sizePerLocalJ;
    int localIdx = localIJIndex + (threadIdx.x + 1);

    s_data[localIdx] = B[globalBIdx];

    if (threadIdx.z == 0)
    {
        s_data[localIdx - sizePerLocalI] = B[globalBIdx - sizePerGlobalBI];
        s_data[localIdx + blockDim.z * sizePerLocalI] = B[globalBIdx + blockDim.z * sizePerGlobalBI];
    }
    if (threadIdx.y == 0)
    {
        s_data[localIdx - sizePerLocalJ] = B[globalBIdx - sizePerGlobalBJ];
        s_data[localIdx + blockDim.y * sizePerLocalJ] = B[globalBIdx + blockDim.y * sizePerGlobalBJ];
    }
    if (threadIdx.x == 0)
    {
        s_data[localIdx - 1] = B[globalBIdx - 1];
        s_data[localIdx + blockDim.x] = B[globalBIdx + blockDim.x];
    }

    __syncthreads();

    if (globalK == n - 1 || globalJ == n - 1 || globalI == n - 1)
    {
        A[globalIdx] = 0;
    }
    else
    {
        A[globalIdx] = (float)0.8 * (s_data[localIdx - sizePerLocalI] +
                                     s_data[localIdx + sizePerLocalI] +
                                     s_data[localIdx - sizePerLocalJ] +
                                     s_data[localIdx + sizePerLocalJ] +
                                     s_data[localIdx - 1] +
                                     s_data[localIdx + 1]);
    }
}

int main(int argc, char *argv[])
{
    int error = 0;

    /* Get Dimension */
    if (argc != 2)
    {
        printf("Error: The number of arguments is not exactly 1\n");
        return 0;
    }
    int n = atoi(argv[1]);
    size_t numElem = n * n * n;
    size_t numBytes = numElem * sizeof(float);

    int nB = n + 1;
    size_t numElemB = nB * nB * nB;
    size_t numBytesB = numElemB * sizeof(float);

#ifndef NDEBUG
    printf("n=%d, numElem=%ld, numBytes=%ld\n", n, numElem, numBytes);
    printf("nB=%d, numElemB=%ld, numBytesB=%ld\n", nB, numElemB, numBytesB);
#endif

    /* Allocate Host Memory */
    float *h_B = NULL;
    error = error || cudaHostAlloc((void **)&h_B, numBytesB, 0);
#ifndef NDEBUG
    float *h_hA = (float *)malloc(numBytes);
#endif
    float *h_dA = NULL;
    error = error || cudaHostAlloc((void **)&h_dA, numBytes, 0);
    if (error)
    {
        printf("Error: cudaHostAlloc returns error\n");
        return 0;
    }

    /* Initialize Host Memory */
    initB(h_B, nB);
#ifndef NDEBUG
    double timestampPreCpuKernel = getTimeStamp();
    jacobiRelaxationReference(h_hA, h_B, n);
    double timestampPostCpuKernel = getTimeStamp();
    printf("CPU: %lf %ld\n", sumA(h_hA, n), (long)ceil(1000*(timestampPostCpuKernel - timestampPreCpuKernel)));
#endif

    /* Allocate Device Memory */
    float *d_B = NULL;
    error = error || cudaMalloc((void **)&d_B, numBytesB);
    float *d_A = NULL;
    error = error || cudaMalloc((void **)&d_A, numBytes);
    if (error)
    {
        printf("Error: cudaMalloc returns error\n");
        return 0;
    }

    /* Configuration */
#define NUM_STREAM 2
    int nIStreams[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; i++)
    {
        nIStreams[i] = n / NUM_STREAM;
    }
    nIStreams[NUM_STREAM - 1] += n % NUM_STREAM;

    dim3 d_blockDim;
    d_blockDim.x = 32;
    d_blockDim.y = 32;
    d_blockDim.z = 1; // must be 1

    dim3 d_gridDimStreams[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; i++)
    {
        d_gridDimStreams[i].x = (n - 1) / d_blockDim.x + 1;
        d_gridDimStreams[i].y = (n - 1) / d_blockDim.y + 1;
        d_gridDimStreams[i].z = (nIStreams[i] - 1) / d_blockDim.z + 1;
    }

    /* Create NUM_STREAM Streams */
    cudaStream_t d_streams[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; i++)
    {
        error = error || cudaStreamCreate(&d_streams[i]);
    }
    if (error)
    {
        printf("Error: cudaStreamCreate returns error\n");
        return 0;
    }

    // TIMER BEGIN
    /* Copy Host Memory to Device Memory */
    double timestampPreCpuGpuTransfer = getTimeStamp();

    size_t numElemBStream1 = 0;
    if (NUM_STREAM != 1)
    {
        numElemBStream1 = (nIStreams[0] + 1 + 1) * nB * nB;
    }
    else
    {
        numElemBStream1 = (nIStreams[0] + 1) * nB * nB;
    }
    error = error || cudaMemcpyAsync(d_B, h_B, numElemBStream1 * sizeof(float), cudaMemcpyHostToDevice, d_streams[0]);
    if (NUM_STREAM != 1)
    {
        cudaStreamSynchronize(d_streams[0]);
    }

    int numElemBStreams = numElemBStream1;
    for (int i = 1; i < NUM_STREAM; i++)
    {
        int nBIStreami = nIStreams[i];
        size_t numElemBStreami = ((i == NUM_STREAM - 1) ? nBIStreami - 1 : nBIStreami) * nB * nB;
        error = error || cudaMemcpyAsync(d_B + numElemBStreams, h_B + numElemBStreams, numElemBStreami * sizeof(float), cudaMemcpyHostToDevice, d_streams[i]);
        numElemBStreams += numElemBStreami;
        if (i != NUM_STREAM - 1)
        {
            // Synchronize between cudaMemcpyAsync
            cudaStreamSynchronize(d_streams[i]);
        }
    }

    if (numElemBStreams != numElemB)
    {
        printf("Error: cudaMemcpyAsync does not cover entire B (%ld != %ld)\n", numElemBStreams, numElemB);
        return 0;
    }
    if (error)
    {
        printf("Error: cudaMemcpyAsync B returns error %d\n", error);
        return 0;
    }

    /* Run Kernel */
    int d_smemNumElem = (d_blockDim.x + 2) * (d_blockDim.y + 2) * (d_blockDim.z + 2);
    size_t d_smemNumBytes = d_smemNumElem * sizeof(float);
    size_t d_startingI = 0;
    for (int i = 0; i < NUM_STREAM; i++)
    {
        jacobiRelaxation<<<d_gridDimStreams[i], d_blockDim, d_smemNumBytes, d_streams[i]>>>(d_A, d_B, n, d_startingI);
        d_startingI += nIStreams[i];
    }

    /* Copy Device Memory to Host Memory */
    size_t numElemAStreams = 0;
    for (int i = 0; i < NUM_STREAM; i++)
    {
        size_t numElemAStreami = nIStreams[i] * n * n;
        error = error || cudaMemcpyAsync(h_dA + numElemAStreams, d_A + numElemAStreams, numElemAStreami * sizeof(float), cudaMemcpyDeviceToHost, d_streams[i]);
        numElemAStreams += numElemAStreami;
    }

    if (numElemAStreams != numElem)
    {
        printf("Error: cudaMemcpyAsync does not cover entire A\n");
        return 0;
    }
    if (error)
    {
        printf("Error: cudaMemcpyAsync A returns error %d\n", error);
        return 0;
    }

    /* Synchronize Streams */
    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamSynchronize(d_streams[i]);
    }

    double timestampPostGpuCpuTransfer = getTimeStamp();
    // TIMER END

    /* Free Device Memory */
    cudaFree(d_A);
    d_A = NULL;
    cudaFree(d_B);
    d_B = NULL;

    /* Output */
    double aValue = sumA(h_dA, n);
    long totalGpuElapased = (long)ceil(1000*(timestampPostGpuCpuTransfer - timestampPreCpuGpuTransfer));
    printf("%lf %ld\n", aValue, totalGpuElapased);
    
#ifndef NDEBUG
    for (int i = 0; i < NUM_STREAM; i++)
    {
        printf("d_gridDimStream%d=(%d, %d, %d), d_blockDim=(%d, %d, %d), d_smemNumBytes=%ld\n", i, d_gridDimStreams[i].x, d_gridDimStreams[i].y, d_gridDimStreams[i].z, d_blockDim.x, d_blockDim.y, d_blockDim.z, d_smemNumBytes);
    }

    /* Verify Device Result with Host Result */
    error = error || !checkA(h_hA, h_dA, n);
    if(error)
    {
        printf("Error: GPU result does not with CPU result\n");
    }
#endif
    
    /* Free Host Memory */
    cudaFreeHost(h_dA);
    h_dA = NULL;
#ifndef NDEBUG
    free(h_hA);
    h_hA = NULL;
#endif
    cudaFreeHost(h_B);
    h_B = NULL;

    /* Clean Up Device Resource */
    cudaDeviceReset();
}
