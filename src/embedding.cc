#include "embedding.h"
#include <exception>
#include <mkl.h>
#include <iostream>
#include <memory.h>

namespace lh
{

template <class T>
Embedding<T>::Embedding(std::vector<std::string> names, Graph<T> &pb_graph)
{
    /*整体概述
        嵌入层是深度学习中常用的模块，用于将离散标记转化为连续的嵌入表示。
        这段代码提供了一个通用的模板类 Embedding，支持动态类型（通过模板参数 T）。
        它包括初始化（加载嵌入权重）、计算嵌入表示（通过索引查找）和析构（释放内存）。
    */
    if (names.size() > 1)//确保 names 的大小为 1，嵌入层只需要一个权重参数。
        throw std::invalid_argument("embedding only need 1 arg!");
    std::string name_w = names[0];
    if (pb_graph.find(name_w) == pb_graph.end())//根据 name_w 查找对应的权重 w（w.first：权重的维度信息，w.second：权重数据）
        throw std::invalid_argument("name " + name_w + " not found in graph!");
    Param<T>& w = pb_graph[name_w];
    std::vector<std::size_t> dims = w.first;
    
    //从模型中提取出权重维度信息 和 权重数据信息
    vocab_size_ = dims[0];//词汇表的大小（行数）
    embedding_size_ = dims[1];//每个嵌入向量的维度（列数）
    
    weight = new T[vocab_size_ * embedding_size_];
    for (std::size_t i = 0; i < vocab_size_ * embedding_size_; i++)
    {
        weight[i] = w.second[i];
    }
}

template <class T>
Embedding<T>::~Embedding()
{
    delete[] weight;
}

template <class T>
void Embedding<T>::compute(std::size_t batch_size, std::size_t seq_len, uint64_t *input, T *output)
{   //嵌入层实现了从离散标记到连续向量的映射功能，是嵌入层计算的核心部分
    for (std::size_t i = 0; i < batch_size * seq_len; i++)
    {
        T *start = output + i * embedding_size_; //计算当前输出嵌入向量的起始位置 start
        uint64_t index = input[i]; //遍历每个输入标记索引 index
        if (index >= vocab_size_) //如果索引超出词汇表大小，则抛出异常。
            throw std::invalid_argument("index must small than vocab size");
        T *weight_start = weight + index * embedding_size_; //根据索引 index，找到对应的嵌入向量起始位置 weight_start
        memcpy(start, weight_start, embedding_size_ * sizeof(T)); //使用 memcpy 将嵌入向量从 weight_start 复制到 start：
    }
}

template class Embedding<float>;
} // namespace lh
