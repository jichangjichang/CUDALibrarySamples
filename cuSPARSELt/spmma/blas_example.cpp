/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cublas.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <iostream>            // std::rand
#include <iomanip>
#include <cmath>
#include <vector>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                       \
{                                                                              \
    cublasStatus_t status = (func);                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \
        printf("CUBLSE API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString((cudaError_t)status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#define MATRIX_K WIDTH
//#define VALIDATE TRUE
constexpr int EXIT_UNSUPPORTED = 2;

int run_blas();

int main(void) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (major_cc < 8) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability >= 8.0, current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    run_blas();

    return EXIT_SUCCESS;
}

int run_blas()
{
    // Host problem definition, row-major order
    //
    //
std::printf("Trans     m     n     k   TFLOPS\n");
for(int tn_nn_count = 0; tn_nn_count< 2 ; tn_nn_count++)
{

int max_value = 20480;
int step;
std::vector<int> mn_size;
std::vector<int> k_size;
auto          opA   = CUBLAS_OP_T;


if(tn_nn_count == 0) //tn
{
step = 1280;
mn_size.push_back(10240);
k_size.push_back(2560);
for(int k_init = k_size[0]+step; k_init<=max_value;k_init+=step)
	k_size.push_back(k_init);
}
else //nn
{
opA   = CUBLAS_OP_N;
step = 1024;
mn_size.push_back(3072);
k_size.push_back(10240);
for(int mn_init = mn_size[0]+step; mn_init<=max_value;mn_init+=step)
	mn_size.push_back(mn_init);
}
for(auto itrMN : mn_size){
for(auto itrK : k_size){
    int m     = itrMN; // bigger sizes may require dynamic allocations
    int n     = itrMN; // bigger sizes may require dynamic allocations
    int k     = itrK; // bigger sizes may require dynamic allocations
    auto          opB   = CUBLAS_OP_N;
    auto          type  = CUDA_R_16F;
   // auto          compute_type = CUSPARSE_COMPUTE_16F;

    bool     is_rowmajor    = 0;//(order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUBLAS_OP_N);
    bool     isB_transposed = (opB != CUBLAS_OP_N);
    auto     num_A_rows     = (isA_transposed) ? k : m; //K
    auto     num_A_cols     = (isA_transposed) ? m : k; //m
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = num_A_rows;
    auto     ldb            = num_B_rows;
    auto     ldc            = m;
    auto     A_height       = num_A_cols;
    auto     B_height       = num_B_cols;
    auto     C_height       = n;
    auto     A_size         = A_height * lda * sizeof(__half);
    auto     B_size         = B_height * ldb * sizeof(__half);
    auto     C_size         = C_height * ldc * sizeof(__half);
    auto     Result_C_size         = C_height * ldc * sizeof(float);
    __half *hA,*hB,*hC;
    hA = (__half*)malloc(A_size);    
    hB = (__half*)malloc(B_size);    
    hC = (__half*)malloc(C_size);    
    for (int i = 0; i < m * k; i++)
        hA[i] = static_cast<__half>(static_cast<float>(std::rand() % 10 - 5));
    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<__half>(static_cast<float>(std::rand() % 10 - 5));
    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC, *dD;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )

    //--------------------------------------------------------------------------
    cudaStream_t                   stream = nullptr;
    cudaEvent_t startEvent = NULL, stopEvent = NULL;

    // Create CUDA event to time the execution time of each algo    
    CHECK_CUDA(cudaEventCreate(&startEvent))
    CHECK_CUDA(cudaEventCreate(&stopEvent))

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle))

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH))

    CHECK_CUDA( cudaStreamSynchronize(stream) )
    // Perform the matrix multiplication
    float time, fast_time;
    fast_time = 10000000;

    //std::printf("(m, n, k, lda, ldb, ldc) = (%5d, %5d, %5d, %5d, %5d, %5d)   ", m, n, k,lda,ldb,ldc);
for(int loop = 0;loop < 10; loop++)
{
    CHECK_CUDA(cudaEventRecord(startEvent, stream))
    CHECK_CUBLAS(cublasGemmEx(handle, opA, opB, m, n, k, (void *)&alpha,
                          (void *)dA, CUDA_R_16F, lda,
                          (void *)dB, CUDA_R_16F, ldb,
                          (void *)&beta, (void *)dC, CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) )
    CHECK_CUDA(cudaEventRecord(stopEvent, stream) )
    CHECK_CUDA(cudaEventSynchronize(stopEvent) )
    CHECK_CUDA(cudaEventElapsedTime(&time, startEvent, stopEvent))
    if(time< fast_time)
	fast_time = time;
}
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    time = fast_time;
    double tflops = ((double)2*m*n*k)/(double)time * 1000 / 1000000000000;
    //std::cout << "time = " << time<<" ms"<< std::endl;
    //std::cout << "m = " <<   m <<  std::endl;
    //std::cout << "n = " <<   n <<  std::endl;
    //std::cout << "k = " <<   k <<  std::endl;

    std::printf("%s    %5d %5d %5d   ", (tn_nn_count)?"NN":"TN"  , m, n, k);
    std::cout << std::fixed;
    std::cout << std::setprecision (2)  << tflops << std::endl;
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // device result check
    // matrix A has been pruned
    //CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    //CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);
    int correct = 1;
    // host computation
    float* hC_result;
#ifdef VALIDATE
    hC_result = (float*)malloc(Result_C_size);    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int k1 = 0; k1 < k; k1++) {
                auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                sum      += static_cast<float>(hA[posA]) *  // [i][k]
                            static_cast<float>(hB[posB]);   // [k][j]
            }
            auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            hC_result[posC] = sum;  // [i][j]
        }
    }
//    // host-device comparison
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            auto device_value = static_cast<float>(hC[pos]);
            auto host_value   = hC_result[pos];
            auto tolerence_value   = abs((device_value- host_value)/device_value);
            if (tolerence_value > 0.001) {
                // direct floating point comparison is not reliable
                std::printf("(%d, %d):\t%f vs. %f\n",
                            i, j, host_value, device_value);
                correct = 0;
                break;
            }
        }
    }
    if (correct)
        std::printf("spmma_example test PASSED\n");
    else
        std::printf("spmma_example test FAILED: wrong result\n");
    free((void*)hC_result);
#endif
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUBLAS( cublasDestroy(handle))
    CHECK_CUDA( cudaEventDestroy(startEvent))
    CHECK_CUDA( cudaEventDestroy(stopEvent))
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    free((void*)hA);
    free((void*)hB);
    free((void*)hC);
    //std::printf("free all memory\n");
}// for k
}// for mn

}//tn_nn
 return 1;
}
