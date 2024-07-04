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
        }

        torch::Tensor forward(torch::Tensor x){
            x = transformer->forward(x);
            // logits, shape (B, T, vocab_size)
            return lm_head->forward(x);
        }

        std::vector<std::vector<uint64_t>> inference(const std::vector<uint64_t>& token_encodings, 
                                                     const int num_return_sequences, 
                                                     const int max_generation_length, 
                                                     const torch::Device& device);
};


// Entry function to load the pretrained GPT-2 model,
// and generate language sequence(s) based on a given input sequence.
//
// @param tiktoken_conf             Path to where tiktoken conf file is.
// @param input_string              Input language sequence.
// @param target_sequence_length    Target number of tokens in the output language sequence.
//                                  (i.e., if the input string is long, generated words will be shorter)
// @param num_output_variants       Number of output sequences to generate given the input sequence.
//
void GPT_playground(std::string tiktoken_conf, std::string input_string, int target_sequence_length, int num_output_variants);