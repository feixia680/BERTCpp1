#include "dense.h"
#include <exception>
#include <mkl.h>
#include <memory.h>

namespace lh{

    template<class T>
    //功能：Dense 类实现了一个全连接层的基本操作，包括权重矩阵乘法和偏置向量的加法。
    Dense<T>::Dense(std::vector<std::string> names, Graph<T>& pb_graph){
        std::string name_w = names[0];
        if(pb_graph.find(name_w) == pb_graph.end()) throw std::invalid_argument("name " + name_w + " not found in graph!");
        Param<T>& w = pb_graph[name_w];
        std::vector<std::size_t> dims = w.first; 
        input_size_ = dims[0];
        output_size_ = dims[1];
        weight = new T[input_size_*output_size_];
        for(std::size_t i=0; i<input_size_*output_size_; i++){
            weight[i] = w.second[i];
        }
        
        if(names.size() > 1){ //加载偏置（可选）如果提供了偏置名称，则加载偏置向量。
            std::string name_b = names[1];
            if(pb_graph.find(name_b) == pb_graph.end()) throw std::invalid_argument("name " + name_b +  " not found in graph!");
            Param<T>& b = pb_graph[name_b]; 
            bias = new T[output_size_];
            for(std::size_t i=0; i<output_size_; i++){
                bias[i] = b.second[i];
            }
        }
        else bias = nullptr; //如果没有提供偏置名称，bias 指针将设置为 nullptr

        weight_observer = nullptr;
    }

    template<class T>
    Dense<T>::~Dense(){
        delete [] weight;
        if(bias != nullptr) delete [] bias;
    }

    template<>
    void Dense<float>::multiplyweight(std::size_t batch_size, std::size_t seq_len, float* input, float* output){
        //进行矩阵乘法操作，计算输入数据和权重矩阵的乘积
        //使用 cblas_sgemm（Intel Math Kernel Library 中的一部分）执行高效的矩阵乘法
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size*seq_len, output_size_, input_size_, 1.0f, input, input_size_, weight, output_size_, 0.0f, output, output_size_);
    }

    template<>
    void Dense<float>::multiplyweight2(std::size_t batch_size, std::size_t seq_len, float* input, float* output) {
        // 初始化输出矩阵为零
        std::fill(output, output + (batch_size * seq_len * output_size_), 0);
        // 手动实现矩阵乘法
        for (std::size_t b = 0; b < batch_size; ++b) {
            for (std::size_t s = 0; s < seq_len; ++s) {
                for (std::size_t i = 0; i < output_size_; ++i) {  // 对输出矩阵的每一行
                    float sum = 0.0;
                    for (std::size_t j = 0; j < input_size_; ++j) {  // 对输入矩阵的每一列
                        sum += input[(b * seq_len + s) * input_size_ + j] * weight[i * input_size_ + j];
                    }
                    output[(b * seq_len + s) * output_size_ + i] = sum;
                }
            }
        }
    }


    template<>
    void Dense<float>::addbias(std::size_t batch_size, std::size_t seq_len, float* output){
        //功能：将偏置向量加到每一个输出向量上
        for(std::size_t idx = 0; idx < batch_size * seq_len; idx++){
            for(std::size_t feature_idx = 0; feature_idx < output_size_; feature_idx++){
                output[idx * output_size_ + feature_idx] += bias[feature_idx];
            }
        }
    }

    template<>
    void Dense<float>::compute(std::size_t batch_size, std::size_t seq_len, float* input, float* output){
        // input shape [batch_size, input_size_]
        // output shape [batch_size, output_size_]

        /*
            执行权重乘法。
            如果有偏置，则添加偏置。
        */
        
        multiplyweight(batch_size, seq_len, input, output);
        // add bias vector here
        if(bias != nullptr){
            addbias(batch_size, seq_len, output);
        }
    }

    template<>
    void Dense<float>::addobserver(float average_constant){
        //添加一个观测器，用于监测和调整权重的分布
        weight_observer = new Observer(average_constant);
    }

    template<>
    void Dense<float>::calibration(std::size_t batch_size, std::size_t seq_len, float* input, float* output){
        //校准全连接层的权重，记录输出结果以供观测器分析
        // calibration data, record blas output
        /*
            特别注意：
            必须在添加观测器之后调用校准函数。
        */

        if(weight_observer == nullptr) throw std::invalid_argument("the observer is null, please add observer before calibration!");
        
        multiplyweight(batch_size, seq_len, input, output);
        
        // record output
        std::size_t size = batch_size * seq_len * output_size_;
        weight_observer->compute(output, size);

        // add bias vector here
        if(bias != nullptr){
            addbias(batch_size, seq_len, output);
        }

    }

    template class Dense<float>;

}