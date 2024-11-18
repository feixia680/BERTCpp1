#include "pooler.h"
#include <cmath>
#include <memory.h>

namespace lh
{
    template<>
    void tanh_<float>(std::size_t size, float* input){
        //为浮点数数组中的每个元素应用Tanh激活函数
        for(std::size_t i = 0; i < size; i++) input[i] = tanhf(input[i]);
    }


    template<class T>
    //Pooler类主要用于将BERT的序列输出转换为一个固定长度的句子向量，通常用于分类任务
    
    /*核心处理流程：
        从每个序列的输出中提取第一个token的输出。
        通过全连接层处理这些输出。
        应用Tanh激活函数。
    */
    Pooler<T>::Pooler(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t hidden_size){
        hidden_size_ = hidden_size;//hidden_size：隐藏层的大小，即每个token输出的维度。

        tranfor_dense_ = new Dense<T>(names, pb_graph);//创建一个Dense全连接层对象。

        tranfor_dense_output_ = new T[pre_batch_size * hidden_size];//分配输出数组空间，用于存储全连接层的中间输出。
    }

    template<class T>
    Pooler<T>::~Pooler(){

        delete tranfor_dense_;

        delete [] tranfor_dense_output_;
    }

    template<class T>
    void Pooler<T>::compute(std::size_t batch_size, std::size_t seq_len, T* input, T* output){
        
        for(std::size_t idx = 0; idx < batch_size; idx++){
            //提取每个序列的第一个token的输出
            memcpy(tranfor_dense_output_ + idx * hidden_size_, input + idx * seq_len * hidden_size_, hidden_size_*sizeof(T));
        }

        //调用Dense对象的compute方法处理数据，每个序列处理为一个输出。
        tranfor_dense_->compute(batch_size, 1, tranfor_dense_output_, output);

        //使用tanh_函数激活全连接层的输出，实现非线性变换
        tanh_(batch_size * hidden_size_, output);

    }

    template class Pooler<float>;
    
} // namespace lh
