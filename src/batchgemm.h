#pragma once

#include "util.h"
// 只声明，不初始化
extern std::vector<std::vector<float>> mul_table;
extern float lookup(float i, float j);
namespace lh{
    void printMatrix(const float* matrix, int rows, int cols) ;
    template<class T>
    void attn_qk(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, T* query, T* key, T* output, const T** q_array, const T** k_array, T** pointer_qk_array);

    template<class T>
    void attn_sv(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, T* sim, T* value, T* output, const T** sim_array, const T** value_array, T** pointer_sv_array);
}