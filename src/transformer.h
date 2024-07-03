#pragma once

#include <torch/torch.h>
#include <cmath>
#include <memory>

class AttentionBlock : public torch::nn::Module {
    // Transformer's attention block - communication followed by computation (multihead attention then feedforward)
    public:
        torch::nn::LayerNorm layer_norm1{nullptr};
        torch::nn::LayerNorm layer_norm2{nullptr};

        AttentionBlock(int context_window_size, int embedding_dims, int n_heads, float dropout_prob, int seed_num){
            torch::manual_seed(seed_num);

            layer_norm1 = register_module("layer_norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
            layer_norm2 = register_module("layer_norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
        }

        torch::Tensor forward(torch::Tensor x){
            // Layer norms are applied prior to attention and feedforward.
            auto y1 = layer_norm1->forward(x);
            y1 = x; // + at_heads->forward(y1);

            auto y2 = layer_norm2->forward(y1);
            y2 = y1; // + feed_forward->forward(y2);

            return y2;
        }
};

class Transformer : public torch::nn::Module {
    public:
        torch::nn::Embedding token_embedding{nullptr};
        torch::nn::Embedding position_embedding{nullptr};
        torch::nn::Sequential attn_blocks{nullptr};
        torch::nn::LayerNorm layer_norm{nullptr};

        Transformer(int vocab_size, int context_window_size, int embedding_dims,
                    int n_heads, int n_layers, float dropout_prob = 0.0, int seed_num = 42){
            torch::manual_seed(seed_num);
            token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, embedding_dims));
            position_embedding = register_module("position_embedding", torch::nn::Embedding(context_window_size, embedding_dims));
            attn_blocks = register_module("attention_blocks", torch::nn::Sequential());
            for(size_t i=0; i<n_layers; ++i){
                attn_blocks->push_back(AttentionBlock(context_window_size, embedding_dims, n_heads, dropout_prob, seed_num));
            }
            layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
        }
};