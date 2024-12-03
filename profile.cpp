#include "src/bert.h"
#include "src/tokenizer.h"
#include "src/model.pb.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>  // for min
#include <numeric>    // for accumulate
#include <iomanip>    // for setprecision


using namespace std;
using namespace lh;

const int SIZE = 10000;
std::vector<std::vector<float>> mul_table(SIZE, std::vector<float>(SIZE));
static bool is_initialized = false;

// creat table
void initialize_mul_table() {
    if (!is_initialized) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                mul_table[i][j] = static_cast<float>(i * j); // Convert result to float
            }
        }
        is_initialized = true;
    }
}

// Lookup function for the multiplication table
float lookup(float i, float j) {
    int sign = 1;

    // Check sign
    if ((i < 0 && j > 0) || (i > 0 && j < 0)) {
        sign = -1;
    }
    i = std::abs(i);
    j = std::abs(j);

    if (!is_initialized) {
        initialize_mul_table();
    }
    float result = 0;

    if ((i >= SIZE) && (j >= SIZE)) {
        i /= 10;
        j /= 10;
        result = mul_table[i][j] * 100;
    }
    else if ((i >= SIZE) && (j >= 0 && j < SIZE)) {
        i /= 10;
        result = mul_table[i][j] * 10;
    }
    else if ((j >= SIZE) && (i >= 0 && i < SIZE)) {
        j /= 10;
        result = mul_table[i][j] * 10;
    }
    else if ((i >= 0 && i < SIZE) && (j >= 0 && j < SIZE)) {
        result = mul_table[i][j];
    }
    else {
        throw std::out_of_range("Input values are out of the predefined range");
    }

    return (result * sign) / 100000000.0f;
}

// 打印参数信息的函数
void PrintGraphParameters(const lh::Graph<float>& graph) {
    for (const auto& entry : graph) {
        const std::string& name = entry.first;
        const std::vector<size_t>& dims = entry.second.first;
        const float* data = entry.second.second;

        std::cout << "Parameter Name: " << name << std::endl;

        // Print dimensions
        std::cout << "Dimensions: ";
        for (size_t dim : dims) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "----------------------------------" << std::endl;
    }
}


void logInputIdsToFile(uint64_t * input_ids, std::size_t batch_size, std::size_t seq_len, const std::string& log_file) {
    // 打开日志文件
    std::ofstream logStream(log_file, std::ios::app);
    if (!logStream.is_open()) {
        std::cerr << "Failed to open log file: " << log_file << std::endl;
        return;
    }

    // 写入 input_ids 的内容
    logStream << "Logging input_ids:" << std::endl;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        logStream << "Batch " << batch << ": ";
        for (std::size_t seq = 0; seq < seq_len; ++seq) {
            logStream << input_ids[batch * seq_len + seq] << " ";
        }
        logStream << std::endl;
    }

    logStream.close();
}

int main()
{
    Model model;
    Graph<float> graph;
    fstream input("/workspace/model/model.bin", ios::in | ios::binary);
    if (!model.ParseFromIstream(&input))
    {
        throw std::invalid_argument("can not read protofile");
    }
    for (int i = 0; i < model.param_size(); i++)
    {
        Model_Paramter paramter = model.param(i);
        int size = 1;
        vector<size_t> dims(paramter.n_dim());
        for (int j = 0; j < paramter.n_dim(); j++)
        {
            int dim = paramter.dim(j);
            size *= dim;
            dims[j] = dim;
        }
        float *data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = paramter.data(i);
        }
        graph[paramter.name()] = make_pair(dims, data);
    }
    google::protobuf::ShutdownProtobufLibrary();
    // 打印参数信息
    //PrintGraphParameters(graph);
    cout << "load paramter from protobuf successly!" << endl;

    size_t pre_batch_size = 100;
    size_t pre_seq_len = 512;
    size_t num_heads = 12;
    size_t embedding_size = 768;
    size_t head_hidden_size = 64;
    size_t intermediate_ratio = 4;
    size_t num_layers = 12;
    vector<string> names;
    names.push_back("embeddings.word_embeddings.weight");
    names.push_back("embeddings.position_embeddings.weight");
    names.push_back("embeddings.token_type_embeddings.weight");
    names.push_back("embeddings.LayerNorm.weight");
    names.push_back("embeddings.LayerNorm.bias");
    for (int idx; idx < num_layers; idx++)
    {
        names.push_back("encoder.layer." + to_string(idx) + ".attention.self.query.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.self.query.bias");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.self.key.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.self.key.bias");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.self.value.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.self.value.bias");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.output.dense.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.output.dense.bias");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.bias");
        names.push_back("encoder.layer." + to_string(idx) + ".intermediate.dense.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".intermediate.dense.bias");
        names.push_back("encoder.layer." + to_string(idx) + ".output.dense.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".output.dense.bias");
        names.push_back("encoder.layer." + to_string(idx) + ".output.LayerNorm.weight");
        names.push_back("encoder.layer." + to_string(idx) + ".output.LayerNorm.bias");
    }
    names.push_back("pooler.dense.weight");
    names.push_back("pooler.dense.bias");

    Bert<float> bert(names, graph, pre_batch_size, pre_seq_len, embedding_size, num_heads, head_hidden_size, intermediate_ratio, num_layers);
    FullTokenizer tokenizer("/workspace/model/bert-base-uncased-vocab.txt");

    cout << "init model from pb file and tokenizer successly!" << endl;

    vector<string> input_string = {u8"how are you! i am very happy to see you guys, please give me five ok? thanks", u8"this is some jokes, please tell somebody else that reputation to user privacy protection. There is no central authority or supervisor having overall manipulations over others, which makes Bitcoin favored by many. Unlike filling piles of identity information sheets before opening bank accounts, users of Bitcoin need only a pseudonym, a.k.a an address or a hashed public key, to participate the system."};

    vector<vector<string>> input_tokens(2);
    for (int i = 0; i < 2; i++)
    {
        tokenizer.tokenize(input_string[i].c_str(), &input_tokens[i], 128);
        input_tokens[i].insert(input_tokens[i].begin(), "[CLS]");
        input_tokens[i].push_back("[SEP]");
    }
    uint64_t mask[2];
    for (int i = 0; i < 2; i++)
    {
        mask[i] = input_tokens[i].size();
        for (int j = input_tokens[i].size(); j < 128; j++)
        {
            input_tokens[i].push_back("[PAD]");
        }
    }
    uint64_t input_ids[256];
    uint64_t position_ids[256];
    uint64_t type_ids[256];
    for (int i = 0; i < 2; i++)
    {
        tokenizer.convert_tokens_to_ids(input_tokens[i], input_ids + i * 128);
        for (int j = 0; j < 128; j++)
        {
            position_ids[i * 128 + j] = j;
            type_ids[i * 128 + j] = 0;
        }
    }

    float out[2 * 128 * embedding_size];
    float pool_out[2 * embedding_size];

    //logInputIdsToFile(input_ids, 2, 128, "input_ids.log");  // 打印到日志文件
    auto begin = chrono::system_clock::now();
    for(int i = 0; i < 1; i++) bert.compute(2, 128, input_ids, position_ids, type_ids, mask, out, pool_out);
    auto end = chrono::system_clock::now();
    
    cout<<"Total cpp BERT inference time for 1 run: "<<chrono::duration_cast<chrono::milliseconds>(end-begin).count() <<endl;
}