#pragma once

#include <torch/torch.h>
#include <cmath>
#include <memory>


class Transformer : public torch::nn::Module {
    public:
        torch::nn::Embedding token_embedding{nullptr};
        torch::nn::Embedding position_embedding{nullptr};
        torch::nn::Linear lm_head{nullptr};

        Transformer(int vocab_size, int context_window_size, int embedding_dims,
                    int n_heads, int n_layers, float dropout_prob = 0.0, int seed_num = 42){
            torch::manual_seed(seed_num);
            token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, embedding_dims));
            position_embedding = register_module("position_embedding", torch::nn::Embedding(context_window_size, embedding_dims));

            lm_head = register_module("linear_head", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, vocab_size).bias(false)));
        }
};