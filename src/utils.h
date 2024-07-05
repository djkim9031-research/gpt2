#pragma once
#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>

#include "sw/tokenizer/tiktoken.h"

namespace utils {
    // Extract the layer index from "h.layer_idx" form in hugging face weight pt file.
    inline int extract_layer_num(const std::string& input_string) {
        // Define a regular expression to find the integer after "h."
        std::regex regex_pattern("h\\.([0-9]+)\\.");

        // Initialize a match object
        std::smatch match;

        // Search for the integer using regex
        if (std::regex_search(input_string, match, regex_pattern)) {
            // match[1] contains the first captured group, which is the integer as a string
            std::string integer_str = match[1].str();

            // Convert the captured string to an integer
            int integer_value = std::stoi(integer_str);

            return integer_value;
        } else {
            // Handle case where no match is found (return an error code or throw an exception)
            throw std::invalid_argument("No integer found after 'h.' in the input string.");
        }
    }

    // Helper function to check if a string ends with a specific suffix
    inline bool ends_with(const std::string& fullstring, const std::string& ending) {
        if (fullstring.length() >= ending.length()) {
            return (fullstring.substr(fullstring.length() - ending.length()) == ending);
        } else {
            return false;
        }
    }

    // Set seed function for reproduciblity.
    inline void set_seed(int seed_num){
        torch::manual_seed(seed_num);
        if(torch::cuda::is_available()){
            torch::cuda::manual_seed(seed_num);
        }
    }
}

namespace tokenizer {
    // Tiktoken tokenizer helper class namespace

    struct tiktoken{
        std::unique_ptr<sw::tokenizer::TiktokenFactory> tiktoken_factory;
        std::unique_ptr<sw::tokenizer::Tiktoken> worker;

        tiktoken(std::string tiktoken_conf, std::string encoding_scheme = "p50k_base"){
            tiktoken_factory = std::make_unique<sw::tokenizer::TiktokenFactory>(tiktoken_conf);
            worker = std::make_unique<sw::tokenizer::Tiktoken>(tiktoken_factory->create(encoding_scheme));
        }

        // Tiktoken encode logic.
        std::vector<int64_t> encode(const std::string& input_string) const{
            // Encoding sanity check
            if (worker->decode(worker->encode(input_string)) != input_string) {
                std::cerr << "failed to test tiktoken encode and decode" << std::endl;
                return {};
            }

            std::vector<uint64_t> encoded_uint64 = worker->encode(input_string);
            std::vector<int64_t> encoded(encoded_uint64.begin(), encoded_uint64.end());
            return encoded;
        }

        // Tiktoken decode logic.
        std::string decode(const std::vector<uint64_t>& encoded_tokens) const{
            return worker->decode(encoded_tokens);
        }
    };
}


namespace preprocessing {

    // Preprocessing function - parse the data from the given .txt
    // and returns the tiktoken encoding tensors.
    inline torch::Tensor data_parser(const std::string& data_path,
                                     const tokenizer::tiktoken& tokenizer,
                                     const torch::Device& device){
        
        // Read the content of the file
        std::ifstream file(data_path);
        if (!file.is_open()) {
            std::cerr << "Could not open the file!" << std::endl;
            return torch::Tensor();
        }

        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        // Encode the strings to tokens.
        std::vector<int64_t> encoded_vec = tokenizer.encode(text);
        torch::Tensor tokens = torch::tensor(encoded_vec, torch::kLong).to(device);

        return tokens;
    }

    // Preprocessing function - splitting given data into train/val dataset
    inline void split_dataset(const float data_split_ratio,
                              const torch::Tensor& data,
                              torch::Tensor& train_data,
                              torch::Tensor& val_data){
        
        int n_train = static_cast<int>(data_split_ratio * data.size(0));
        int n_val = data.size(0) - static_cast<int>(data_split_ratio * data.size(0));

        std::vector<torch::Tensor> splits = torch::split(data, {n_train, n_val}, 0);

        train_data = splits[0];
        val_data = splits[1];

        return;
    }

    // Preprocessing function - creating batches of dataset
    inline void create_batch(const int batch_size,
                             const int context_win_size,
                             const torch::Tensor& data,
                             torch::Tensor& batch_input,
                             torch::Tensor& batch_output){
        
        auto idx = torch::randint(0, data.size(0) - context_win_size, {batch_size}, torch::kInt64);

        // Extract the tensors at the selected index for each batch
        std::vector<torch::Tensor> x_vec, y_vec;
        for(size_t i=0; i<batch_size; ++i){
            int start_idx = idx[i].item<int>();
            x_vec.push_back(data.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/start_idx + context_win_size));
            y_vec.push_back(data.slice(/*dim=*/0, /*start=*/start_idx + 1, /*end=*/start_idx + context_win_size + 1));
        }

        // Torch-ify the tensors to form [batch_size, context_win_size] tensors
        batch_input = torch::stack(x_vec);
        batch_output = torch::stack(y_vec);
    }
}

namespace pretrained {

    // Load the pretrained GPT model weights (downloaded from HuggingFace)
    // For linear layers in the attention block (e.g., qkv, projections, mlp layers),
    // the implementation is of shape [output size, input size], whereas for HF downloaded weights 
    // have the shape [input size, output size]. So the corresponding weights need to be transposed.
    inline void load_from_pretrained_GPT2_HF(GPT& model, const std::string& weights_path){
        // Load the saved weights
        torch::jit::script::Module modules = torch::jit::load(weights_path);
        auto parameters = modules.named_parameters();

        // Map the parameter names in C++ GPT model implementation to those in the saved state_dict.
        {
            // Since the C++ GPT model layers are enabled with "requires_grad" field but the loaded weights do not have grads.
            // And this is used for inference.
            torch::NoGradGuard no_grad;
            for(const auto& param : parameters){
                std::string name = param.name;
                auto tensor = param.value;

                if(utils::ends_with(name, "transformer.wte.weight")){
                    model.transformer->token_embedding->weight.copy_(tensor);

                } else if(utils::ends_with(name, "transformer.wpe.weight")){
                    model.transformer->position_embedding->weight.copy_(tensor);

                } else if(utils::ends_with(name, "lm_head.weight")){
                    model.lm_head->weight.copy_(tensor);

                } else if(name.find("transformer.ln_f") != std::string::npos){
                    if(utils::ends_with(name, "weight")){
                        model.transformer->layer_norm->weight.copy_(tensor);

                    } else if(utils::ends_with(name, "bias")){
                        model.transformer->layer_norm->bias.copy_(tensor);

                    } else{
                        std::cerr << "Provided tensor, `"<<name<<"` do not pertain to the transformer.ln_f layer" << std::endl;
                        return;
                    }
                } else if(name.find("transformer.h") != std::string::npos){
                    // Handle weights for attention blocks
                    int curr_layer = utils::extract_layer_num(name);
                    auto curr_attention_block = model.transformer->attn_blocks->ptr<AttentionBlock>(curr_layer);
                    std::string curr_block_prefix = "transformer.h." + std::to_string(curr_layer);

                    if(name.find(curr_block_prefix+".ln_1") != std::string::npos){
                        // layer_norm 1
                        if(utils::ends_with(name, "weight")){
                            curr_attention_block->layer_norm1->weight.copy_(tensor);
    
                        } else if(utils::ends_with(name, "bias")){
                            curr_attention_block->layer_norm1->bias.copy_(tensor);

                        } else{
                            std::cerr << "Provided tensor, `"<<name<<"` do not pertain to the ln_1 layer" << std::endl;
                            return;
                        }
                    } else if(name.find(curr_block_prefix+".ln_2") != std::string::npos){
                        // layer_norm 2
                        if(utils::ends_with(name, "weight")){
                            curr_attention_block->layer_norm2->weight.copy_(tensor);

                        } else if(utils::ends_with(name, "bias")){
                            curr_attention_block->layer_norm2->bias.copy_(tensor);

                        } else{
                            std::cerr << "Provided tensor, `"<<name<<"` do not pertain to the ln_2 layer" << std::endl;
                            return;
                        }
                    } else if(name.find(curr_block_prefix+".attn.c_attn") != std::string::npos){
                        // multihead self-attention qkv
                        if(utils::ends_with(name, "weight")){
                            curr_attention_block->c_attn->c_qkv->weight.copy_(tensor.transpose(0, 1));

                        } else if(utils::ends_with(name, "bias")){
                            curr_attention_block->c_attn->c_qkv->bias.copy_(tensor);

                        } else{
                            std::cerr << "Provided tensor, `"<<name<<"` do not pertain to the attn.c_attn layer" << std::endl;
                            return;
                        }
                    } else if(name.find(curr_block_prefix+".attn.c_proj")!= std::string::npos){
                        // multihead self-attention projection layer
                        if(utils::ends_with(name, "weight")){
                            curr_attention_block->c_attn->c_proj->weight.copy_(tensor.transpose(0, 1));

                        } else if(utils::ends_with(name, "bias")){
                            curr_attention_block->c_attn->c_proj->bias.copy_(tensor);

                        } else{
                            std::cerr << "Provided tensor, `"<<name<<"` do not pertain to the attn.c_proj layer" << std::endl;
                            return;
                        }
                    } else if(name.find(curr_block_prefix+".mlp.c_fc")!=std::string::npos){
                        // mlp fully-connected layer
                        if(utils::ends_with(name, "weight")){
                            curr_attention_block->mlp->c_fc->weight.copy_(tensor.transpose(0, 1));

                        } else if(utils::ends_with(name, "bias")){
                            curr_attention_block->mlp->c_fc->bias.copy_(tensor);

                        } else{
                            std::cerr << "Provided tensor, `"<<name<<"` do not pertain to the mlp.c_fc layer" << std::endl;
                            return;
                        }
                    } else if(name.find(curr_block_prefix+".mlp.c_proj")!=std::string::npos){
                        // mlp projection layer
                        if(utils::ends_with(name, "weight")){
                            curr_attention_block->mlp->c_proj->weight.copy_(tensor.transpose(0, 1));

                        } else if(utils::ends_with(name, "bias")){
                            curr_attention_block->mlp->c_proj->bias.copy_(tensor);
    
                        } else{
                            std::cerr << "Provided tensor, `"<<name<<"` do not pertain to the mlp.c_proj layer" << std::endl;
                            return;
                        }
                    } else{
                        std::cerr << "Provided tensor, `"<<name<<"` do not pertain to any of the implemented C++ model" << std::endl;
                        return;
                    }

                }
            }
        }

        std::cout<<"[INFO]  Pretrained weights loaded successfully."<<std::endl;
    }

}