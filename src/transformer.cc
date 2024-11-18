#include "transformer.h"
#include "memory.h"

#ifdef PRFILE_FUNCTION
    #include <chrono>
    #include <iostream>
#endif

namespace lh{

    template<class T>
    Transformer<T>::Transformer(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t num_heads, std::size_t head_hidden_size, std::size_t intermediate_ratio, std::size_t num_layers){
        //该类负责执行Transformer模型中的各个子层，这些子层协同工作实现复杂的序列到序列或序列到向量的变换
        //Transformer架构广泛应用于NLP任务，如机器翻译、文本生成、语义理解等
        /*参数含义
            names：包含必要参数名称的向量。
            pb_graph：计算图，存储模型参数。
            pre_batch_size、pre_seq_len：预设的批量大小和序列长度。
            num_heads：多头注意力机制中的头数。
            head_hidden_size：每个头的维度。
            intermediate_ratio：中间层大小与隐藏层大小的比例。
            num_layers：Transformer模型的层数。
        */
        std::size_t hidden_size = num_heads * head_hidden_size; //计算隐藏层大小
        std::size_t intermediate_size = hidden_size * intermediate_ratio; //计算中间层大小

        num_heads_ = num_heads;
        hidden_size_ = hidden_size;
        intermediate_size_ = intermediate_size;
        num_layers_ = num_layers;

        //初始化存储每个子层输出的数组。？？？
        mutiheadselfattn_.reserve(num_layers);
        attention_output_dense_.reserve(num_layers);
        attention_layer_norm_.reserve(num_layers);
        intermediate_dense_.reserve(num_layers);
        intermediate_act_.reserve(num_layers);
        output_dense_.reserve(num_layers);
        output_layer_norm_.reserve(num_layers);

        atten_output_.reserve(num_layers);
        atten_dense_output_.reserve(num_layers);
        intermediate_dense_output_.reserve(num_layers);
        output_dense_output_.reserve(num_layers);

        //循环创建每个Transformer层的各个组件（多头自注意力、全连接层、激活函数、层归一化），并分配必要的内存空间。
        auto startit = names.begin();
        for(std::size_t layer_idx = 0; layer_idx < num_layers; layer_idx++){
            std::vector<std::string> attennames(startit, startit+6);
            mutiheadselfattn_[layer_idx] = new MutiheadselfAttn<T>(attennames, pb_graph, pre_batch_size, pre_seq_len, num_heads, head_hidden_size);
            atten_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*hidden_size];
            startit += 6;

            std::vector<std::string> attendensenames(startit, startit+2);
            attention_output_dense_[layer_idx] = new Dense<T>(attendensenames, pb_graph);
            atten_dense_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*hidden_size];
            startit += 2;

            std::vector<std::string> attennormnames(startit, startit+2);
            attention_layer_norm_[layer_idx] = new Layernorm<T>(attennormnames, pb_graph, pre_batch_size, pre_seq_len);
            startit += 2;

            std::vector<std::string> mediatedensenames(startit, startit+2);
            intermediate_dense_[layer_idx] = new Dense<T>(mediatedensenames, pb_graph);
            intermediate_dense_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*intermediate_size];
            startit += 2;

            intermediate_act_[layer_idx] = new Gelu<T>;

            std::vector<std::string> outputdensenames(startit, startit+2);
            output_dense_[layer_idx] = new Dense<T>(outputdensenames, pb_graph);
            output_dense_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*hidden_size];
            startit += 2;

            std::vector<std::string> outputnormnames(startit, startit+2);
            output_layer_norm_[layer_idx] = new Layernorm<T>(outputnormnames, pb_graph, pre_batch_size, pre_seq_len);
            startit += 2;

        } 

    }

    template<class T>
    Transformer<T>::~Transformer(){
        
        for(std::size_t layer_idx = 0; layer_idx < num_layers_; layer_idx++){
            delete mutiheadselfattn_[layer_idx];
            delete attention_output_dense_[layer_idx];
            delete attention_layer_norm_[layer_idx];
            delete intermediate_dense_[layer_idx];
            delete intermediate_act_[layer_idx];
            delete output_dense_[layer_idx];
            delete output_layer_norm_[layer_idx];

            delete atten_output_[layer_idx];
            delete atten_dense_output_[layer_idx];
            delete intermediate_dense_output_[layer_idx];
            delete output_dense_output_[layer_idx];
        }
    }

    template<class T>
    void Transformer<T>::compute(std::size_t batch_size, std::size_t seq_len, T* input, uint64_t* mask, T* output){

        T* pre_input = input;
        /*功能实现：
            多头自注意力层：对输入数据应用多头自注意力。
            前馈全连接层：对自注意力的输出应用全连接层。
            激活函数：对全连接层的输出应用激活函数（GELU）。
            层归一化和残差连接：应用层归一化，并添加自注意力和全连接层的输出到原始输入的残差连接。
            中间层和输出处理：中间层处理后，再次应用全连接层和层归一化，最终将处理结果作为下一层的输入或最终输出。
        */

        for(std::size_t layer_idx = 0; layer_idx < num_layers_; layer_idx++){
            
            #ifdef PRFILE_FUNCTION
                auto begin = std::chrono::system_clock::now();
            #endif

            mutiheadselfattn_[layer_idx]->compute(batch_size, seq_len, pre_input, mask, atten_output_[layer_idx]);

            #ifdef PRFILE_FUNCTION
                auto end = std::chrono::system_clock::now();
                std::cout<<"mutihead attention use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            attention_output_dense_[layer_idx]->compute(batch_size, seq_len, atten_output_[layer_idx], atten_dense_output_[layer_idx]);

            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"attention output dense use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            for(std::size_t idx = 0; idx < batch_size * seq_len * hidden_size_; idx++){
                atten_dense_output_[layer_idx][idx] += pre_input[idx];
            }

            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"attention output shortcut use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif
            
            attention_layer_norm_[layer_idx]->compute(batch_size, seq_len, atten_dense_output_[layer_idx], atten_dense_output_[layer_idx]);
            
            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"attention layernorm use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            intermediate_dense_[layer_idx]->compute(batch_size, seq_len, atten_dense_output_[layer_idx], intermediate_dense_output_[layer_idx]);
            
            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"intermediate dense use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            intermediate_act_[layer_idx]->compute(batch_size*seq_len*intermediate_size_, intermediate_dense_output_[layer_idx]);

            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"intermediate gelu use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            output_dense_[layer_idx]->compute(batch_size, seq_len, intermediate_dense_output_[layer_idx], output_dense_output_[layer_idx]);
            
            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"output dense use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            for(std::size_t idx = 0; idx < batch_size * seq_len * hidden_size_; idx++){
                output_dense_output_[layer_idx][idx] += atten_dense_output_[layer_idx][idx];
            }

            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"output dense short cut use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            output_layer_norm_[layer_idx]->compute(batch_size, seq_len, output_dense_output_[layer_idx], output_dense_output_[layer_idx]);

            #ifdef PRFILE_FUNCTION
                end = std::chrono::system_clock::now();
                std::cout<<"output layernorm use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
                begin = std::chrono::system_clock::now();
            #endif

            pre_input = output_dense_output_[layer_idx];
        }

        memcpy(output, pre_input, sizeof(T)*batch_size*seq_len*hidden_size_);
        //循环结束后，最后一层的输出复制到函数的输出参数 output 中，作为整个 Transformer 网络的最终输出
    }

    template class Transformer<float>;
}