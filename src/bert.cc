#include "bert.h"

#ifdef PRFILE_FUNCTION
    #include <chrono>
    #include <iostream>
#endif

namespace lh{

    template<class T>
    Bert<T>::Bert(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t embedding_size, std::size_t num_heads, std::size_t head_hidden_size, std::size_t intermediate_ratio, std::size_t num_layers){
        //pb_graph：计算图对象，管理所有模型操作。
        //在构造bert的时候 需要构造嵌入层、transformer层、池化层——embednames、transnames后续俩接一下到底是什么
        embedding_size_ = embedding_size;
        hidden_size_ = num_heads * head_hidden_size;
        //embedding_size_ 是嵌入层的维度。
        //hidden_size_ 是隐藏层的维度，计算方式为：注意力头数量 × 每个头的隐藏维度。

        //嵌入层 (bertembedding_)
        //嵌入层的任务是处理标记、位置、类型等输入，生成嵌入表示
        auto startit = names.begin();
        std::vector<std::string> embednames(startit, startit+5);
        bertembedding_ = new BertEmbedding<T>(embednames, pb_graph, pre_batch_size, pre_seq_len, embedding_size);
        embedding_output_ = new T[pre_batch_size * pre_seq_len * embedding_size];
        startit += 5;

        //从 names 中的下一部分取出名字，用于 Transformer 的初始化。 这一块不是很懂
        //16*num_layers 表示每层 Transformer 需要 16 个操作。
        std::vector<std::string> transnames(startit, startit + 16*num_layers);
        transformer_ = new Transformer<T>(transnames, pb_graph, pre_batch_size, pre_seq_len, num_heads, head_hidden_size, intermediate_ratio, num_layers);
        startit += 16*num_layers;

        //池化层
        std::vector<std::string> poolnames(startit, startit + 2);
        pooler_ = new Pooler<T>(poolnames, pb_graph, pre_batch_size, hidden_size_);
        startit += 2;

    }

    template<class T>
    Bert<T>::~Bert(){//析构函数，delete embedding层、transformer层、pooler层
        
        delete bertembedding_;
        delete transformer_;
        delete pooler_;

        delete [] embedding_output_;
    }    

    template<class T> //bert模型的前向计算：嵌入层、transformer 层、池化层计算
    void Bert<T>::compute(std::size_t batch_size, std::size_t seq_len, uint64_t* token_input, uint64_t* posit_input, uint64_t* type_input, uint64_t* mask, T* seq_output, T* pool_output){
        //batch_size 和 seq_len：运行时的批量大小和序列长度。
        //token_input：标记输入
        //posit_input：位置输入
        //type_input：类型输入
        //mask：注意力掩码
        //seq_output：序列输出的存储指针 ？
        //pool_output：池化输出的存储指针
        #ifdef PRFILE_FUNCTION
            auto begin = std::chrono::system_clock::now();
        #endif

        //调用嵌入层
        bertembedding_->compute(batch_size, seq_len, token_input, posit_input, type_input, embedding_output_);

        #ifdef PRFILE_FUNCTION
            auto end = std::chrono::system_clock::now();
            std::cout<<"bert embedding use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
            begin = std::chrono::system_clock::now();
        #endif

        //调用 Transformer 层
        transformer_->compute(batch_size, seq_len, embedding_output_, mask, seq_output);

        #ifdef PRFILE_FUNCTION
            begin = std::chrono::system_clock::now();
        #endif

        //调用池化层
        pooler_->compute(batch_size, seq_len, seq_output, pool_output);

        #ifdef PRFILE_FUNCTION
            end = std::chrono::system_clock::now();
            std::cout<<"bert pooler use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
            begin = std::chrono::system_clock::now();
        #endif
    }

    template class Bert<float>;
}