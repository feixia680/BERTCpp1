#include "bertembedding.h"

namespace lh{

    template<class T>
    BertEmbedding<T>::BertEmbedding(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t embedding_size){
        /* 参数含义
            names：一个字符串列表，用于表示每个嵌入层及层归一化的名称。
            pb_graph：计算图对象，用于管理模型操作。
            pre_batch_size：初始化时的批量大小。
            pre_seq_len：初始化时的序列长度。
            embedding_size：嵌入维度。
        */
        
        /*BertEmbedding 负责将 BERT 模型的输入（标记、位置、类型）转化为一个统一的嵌入表示
            单词嵌入：将每个标记映射到一个嵌入向量。
            位置嵌入：为每个位置添加位置信息的嵌入向量。
            类型嵌入：为每个输入标记添加类型信息的嵌入向量。
            加法融合：将上述三种嵌入相加，形成最终的嵌入。
            归一化处理：通过层归一化（LayerNorm）提高稳定性。
        */
        pre_batch_size_ = pre_batch_size;
        pre_seq_len_ = pre_seq_len;
        embedding_size_ = embedding_size;

        auto startit = names.begin();

        std::vector<std::string> wordembednames(startit, startit+1);//从 names 的第一部分提取名称，初始化 word_embedding_ 对象。
        word_embedding_ = new Embedding<T>(wordembednames, pb_graph);
        word_embedding_output_ = new T[pre_batch_size*pre_seq_len*embedding_size];//分配内存空间 word_embedding_output_ 存储单词嵌入的结果
        startit += 1;

        std::vector<std::string> posiembednames(startit, startit+1);
        position_embedding_ = new Embedding<T>(posiembednames, pb_graph);
        position_embedding_output_ = new T[pre_batch_size*pre_seq_len*embedding_size];
        startit += 1;

        std::vector<std::string> typeembednames(startit, startit+1);
        token_type_embedding_ = new Embedding<T>(typeembednames, pb_graph);
        token_type_embedding_output_ = new T[pre_batch_size*pre_seq_len*embedding_size];
        startit += 1;

        //层归一化用于对嵌入结果进行归一化处理。
        std::vector<std::string> normnames(startit, startit+2);
        embedding_layer_norm_ = new Layernorm<T>(normnames, pb_graph, pre_batch_size, pre_seq_len);
        startit += 2;

    }

    template<class T>
    BertEmbedding<T>::~BertEmbedding(){

        delete word_embedding_;
        delete position_embedding_;
        delete token_type_embedding_;

        delete [] word_embedding_output_;
        delete [] position_embedding_output_;
        delete [] token_type_embedding_output_;
    }

    template<class T>
    void BertEmbedding<T>::compute(std::size_t batch_size, std::size_t seq_len, uint64_t* token_input, uint64_t* posit_input, uint64_t* type_input, T* output){

        word_embedding_->compute(batch_size, seq_len, token_input, word_embedding_output_);
        position_embedding_->compute(batch_size, seq_len, posit_input, position_embedding_output_);
        token_type_embedding_->compute(batch_size, seq_len, type_input, token_type_embedding_output_);

        //遍历所有嵌入向量的元素，将单词、位置和类型嵌入相加，结果保存在 word_embedding_output_ 中
        for(std::size_t j = 0; j < batch_size * seq_len * embedding_size_; j++){
            word_embedding_output_[j] += position_embedding_output_[j] + token_type_embedding_output_[j];
        }

        embedding_layer_norm_->compute(batch_size, seq_len, word_embedding_output_, output);

    }

    template class BertEmbedding<float>;
}