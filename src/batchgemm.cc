#include "batchgemm.h"
#include "mkl.h"
#include <iostream>
namespace lh{
    void printMatrix(const float* matrix, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    template<>
    void attn_qk<float>(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float* query, float* key, float* output, const float** q_array, const float** k_array, float** pointer_qk_array) {

        for (std::size_t idx = 0; idx < batch_size; idx++)
        {
            for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++)
            {
                q_array[idx * num_heads + head_idx] = query + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                k_array[idx * num_heads + head_idx] = key + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                pointer_qk_array[idx * num_heads + head_idx] = output + idx * num_heads * seq_len * seq_len + head_idx * seq_len;
            }
        }
            // 打印所有Query和Key矩阵
        for (std::size_t idx = 0; idx < batch_size * num_heads; ++idx) {
            std::cout << "Query Matrix " << idx + 1 << ":" << std::endl;
            printMatrix(q_array[idx], seq_len, hidden_size);
            std::cout << std::endl;

            std::cout << "Key Matrix " << idx + 1 << ":" << std::endl;
            printMatrix(k_array[idx], seq_len, hidden_size);
            std::cout << std::endl;
        }
        CBLAS_TRANSPOSE tranA = CblasNoTrans;
        CBLAS_TRANSPOSE tranB = CblasTrans;
        const int m = seq_len, n = seq_len, k = hidden_size, lda_array = hidden_size * num_heads, ldb_array = hidden_size * num_heads, ldc_array = seq_len * num_heads, group_size = batch_size * num_heads;
        const float alpha = 1.0, beta = 0.0;
        //此处用到了矩阵乘法
        //cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, q_array, &lda_array, k_array, &ldb_array, &beta, pointer_qk_array, &ldc_array, 1, &group_size);
        
        for (std::size_t idx = 0; idx < batch_size; idx++) {
            for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++) {
                const float* qq = q_array[idx * num_heads + head_idx];
                const float* kk = k_array[idx * num_heads + head_idx];
                float* out = pointer_qk_array[idx * num_heads + head_idx];

                // Print the input matrices q (qq) and k (kk)
                std::cout << "Matrix Q for batch " << idx << ", head " << head_idx << ":\n";
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < k; j++) {  // Printing Q matrix (m x k)
                        std::cout << qq[i * k + j] << " ";
                    }
                    std::cout << std::endl;
                }

                std::cout << "Matrix K for batch " << idx << ", head " << head_idx << ":\n";
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < n; j++) {  // Printing K matrix (k x n)
                        std::cout << kk[i * n + j] << " ";
                    }
                    std::cout << std::endl;
                }

                // Manual matrix multiplication (q * k^T)
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        float sum = 0.0f;
                        for (int l = 0; l < k; l++) {
                            sum += qq[i * k + l] * kk[j * k + l];
                        }
                        out[i * n + j] = alpha * sum + beta * out[i * n + j];
                    }
                }

                // Print the result matrix 'out'
                std::cout << "Result for batch " << idx << ", head " << head_idx << " (q * k^T):\n";
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        std::cout << out[i * n + j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        
       

    }
    

    template<>
    void attn_sv<float>(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float* sim, float* value, float* output, const float** sim_array, const float** value_array, float** pointer_sv_array){

        for (std::size_t idx = 0; idx < batch_size; idx++)
        {
            for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++)
            {
                sim_array[idx * num_heads + head_idx] = sim + idx * num_heads * seq_len * seq_len + head_idx * seq_len;
                value_array[idx * num_heads + head_idx] = value + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                pointer_sv_array[idx * num_heads + head_idx] = output + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
            }
        }
        CBLAS_TRANSPOSE tranA = CblasNoTrans;
        CBLAS_TRANSPOSE tranB = CblasNoTrans;
        const int m = seq_len, n = hidden_size, k = seq_len, lda_array = seq_len * num_heads, ldb_array = hidden_size * num_heads, ldc_array = hidden_size * num_heads, group_size = batch_size * num_heads;
        const float alpha = 1.0, beta = 0.0;
        //此处遇到了矩阵乘法
        cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, sim_array, &lda_array, value_array, &ldb_array, &beta, pointer_sv_array, &ldc_array, 1, &group_size);
        
    }
}