#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Comment out this line to enable debug mode
// #define NDEBUG

/* time stamp function in seconds */
__host__ double getTimeStamp() {
    struct timeval  tv ; gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

__host__ void initX(float* X, int n_rows, int n_cols) {
    for(int i=0; i<n_rows; i++) {
        int ibase = i * n_cols;
        for(int j=0; j<n_cols; j++) {
            X[ibase + j] = (float) (i+j)/2.0;
        }
    }
}

__global__ void f_siggen() {
    // wip
}

int main(int argc, char *argv[]) {
    /* Get Dimension */
    if(argc != 3) {
        printf("Error: The number of arguments is not exactly 2\n");
        return 0;
    }
    int n_rows = atoi(argv[1]);
    int n_cols = atoi(argv[2]);
    int n_elem = n_rows * n_cols;

#ifndef NDEBUG
    printf("n_rows=%d, n_cols=%d, n_elem=%d", n_rows, n_cols, n_elem);
#endif

    /* Allocate and Initialize Host Memory */
    float* h_X = (float*) malloc(sizeof(float) * n_elem);
    float* h_Y = (float*) malloc(sizeof(float) * n_elem);
    float* h_hZ = (float*) malloc(sizeof(float) * n_elem);
    float* h_dZ = (float*) malloc(sizeof(float) * n_elem);

    /* Allocate Device Memory */

    /* Copy Host Memory to Device Memory */

    /* Launch Kernel */
    dim3 gridDim;
    dim3 blockDim;
    size_t d_smemSize = 0;
    f_siggen <<<gridDim, blockDim, d_smemSize>>>();

    /* Copy Device Memory to Host Memory */

    /* Clean Up Device Resource */

    /* Verify Device Result with Host Result */
    int isMatching = 1;

    /* Output */
    if(isMatching) {
#ifndef NDEBUG
        printf("<total_GPU_time> <CPU_GPU_transfer_time> <kernel_time> <GPU_CPU_transfer_time> <Z-value> <nl>\n");
#endif
        float totalGpuElapased = 0;
        float cpuGpuTransferElapsed = 0;
        float kernelElapsed = 0;
        float gpuCpuTransferElapsed = 0;
        float zValue = 0;
        printf("%.6f %.6f %.6f %.6f %.6f\n", totalGpuElapased, cpuGpuTransferElapsed, kernelElapsed, gpuCpuTransferElapsed, zValue);
    } else {
        printf("Error: GPU result does not with CPU result\n");
    }

    /* Clean Up Host Resource */
}