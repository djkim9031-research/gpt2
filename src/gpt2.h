#pragma once
#include <torch/torch.h>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unistd.h>
#include <iostream>
#include <vector>

#include "transformer.h"
#include "sw/tokenizer/tiktoken.h"

struct GPTConfig{
    int context_win_size;
    int vocab_size;
    int n_layers;
    int n_heads;
    int n_embs;

    GPTConfig(int context_win_size = 1024, int vocab_size = 50257, int n_layers = 12, int n_heads = 12, int n_embs = 768)
        : context_win_size(context_win_size), 
          vocab_size(vocab_size),
          n_layers(n_layers),
          n_heads(n_heads),
          n_embs(n_embs) {}
};

class GPT : public torch::nn::Module{
    public:
        GPTConfig config;
        std::shared_ptr<Transformer> transformer{nullptr};
        torch::nn::Linear lm_head{nullptr};
    
        GPT(GPTConfig gpt_config)
            : config(gpt_config){
            
            transformer = register_module("transformer", std::make_shared<Transformer>(config.vocab_size, 
                                                                                       config.context_win_size,
                                                                                       config.n_embs,
                                                                                       config.n_heads,
                                                                                       config.n_layers));
            lm_head = register_module("linear_head", torch::nn::Linear(torch::nn::LinearOptions(config.n_embs, config.vocab_size).bias(false)));

            // Weight sharing (initial token embedding and topmost head layer)
            // The idea is that similar tokens in embedding space should have similar weights contributions (adopted in the original Attention paper).
            // At the top head layer, when tokens are predicted, same weight conribitutions should happen as did in the early token embedding stage.
            // Same memory address are shared, so anything backpropagated via lm_head not only propagates to wte through chain rules, but also directly
            // from lm_head.
            transformer->token_embedding->weight = lm_head->weight;

            // Weight initialization
            // apply is a function provided by torch::nn::Module.
            apply([this](torch::nn::Module& curr_module){
                this->_init_weights(curr_module);
            });
        }

        torch::Tensor forward(torch::Tensor x){
            x = transformer->forward(x);
            // logits, shape (B, T, vocab_size)
            return lm_head->forward(x);
        }

        std::vector<std::vector<uint64_t>> inference(const std::vector<int64_t>& token_encodings, 
                                                     const int num_return_sequences, 
                                                     const int max_generation_length, 
                                                     const torch::Device& device);

    private:

        // Weight initalization scheme
        void _init_weights(torch::nn::Module& curr_module){
            if(auto linear = dynamic_cast<torch::nn::LinearImpl*>(&curr_module)){
                // zero mean, 0.02 std as GPT2 is implemented.
                torch::nn::init::normal_(linear->weight, 0.0, 0.02);
                if(linear->bias.defined()){
                    torch::nn::init::zeros_(linear->bias);
                }
            } else if (auto embedding = dynamic_cast<torch::nn::EmbeddingImpl*>(&curr_module)){
                // zero mean, 0.02 std as GPT2 is implemented.
                torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
            }
        }

};

// Entry function to train GPT-2 model.
// @param data_path                 Path to input data.
// @param tiktoken_conf             Path to where tiktoken conf file is.
// @param gpt_model                 GPT model variants [gpt2, gpt2-medium, gpt2-large, gpt2-xl]
void GPT_trainer(const std::string& data_path, const std::string& tiktoken_conf, const std::string& gpt_model = "gpt2");


// Entry function to load the pretrained GPT-2 model,
// and generate language sequence(s) based on a given input sequence.
//
// @param input_string              Input language sequence.
// @param tiktoken_conf             Path to where tiktoken conf file is.
// @param target_sequence_length    Target number of tokens in the output language sequence.
//                                  (i.e., if the input string is long, generated words will be shorter)
// @param num_output_variants       Number of output sequences to generate given the input sequence.
// @param gpt_model                 GPT model variants [gpt2, gpt2-medium, gpt2-large, gpt2-xl]
//
void GPT_playground(const std::string& input_string, 
                    const std::string& tiktoken_conf, 
                    const int target_sequence_length, 
                    const int num_output_variants, 
                    const std::string& gpt_model = "gpt2");