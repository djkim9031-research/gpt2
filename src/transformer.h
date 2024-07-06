#pragma once

#include <torch/torch.h>
#include <cmath>
#include <memory>

class CausalSelfAttention : public torch::nn::Module {
    // Multihead causal self attention (masked attention) in the transformer decoder
    public:
        torch::nn::Linear c_qkv{nullptr}; //query, key, value projections
        torch::nn::Linear c_proj{nullptr}; //output projections
        int n_heads;
        int n_embs;
        torch::Tensor bias; //Not really a bias, but a lower triangular matrix for masking - following the OpenAI/HF naming convention.

        CausalSelfAttention(int context_window_size, int embedding_dims, int n_heads, float dropout_prob, int seed_num)
            : n_heads(n_heads), n_embs(embedding_dims){
            torch::manual_seed(seed_num);
            c_qkv = register_module("c_qkv", torch::nn::Linear(embedding_dims, 3*embedding_dims));
            c_proj = register_module("c_proj", torch::nn::Linear(embedding_dims, embedding_dims));
            bias = register_buffer("bias", torch::tril(torch::ones({context_window_size, context_window_size})).view({1, 1, context_window_size, context_window_size}));
        }

        torch::Tensor forward(torch::Tensor x){
            auto sizes = x.sizes();
            int B = sizes[0]; // batch size
            int T = sizes[1]; // sequence length
            int C = sizes[2]; // embedding dimension

            // Calculate querys, keys, values for all heads in batch
            auto qkv = c_qkv->forward(x);
            auto q = qkv.slice(/*dim=*/2, /*start idx=*/0, /*end idx=*/n_embs).contiguous();
            auto k = qkv.slice(/*dim=*/2, /*start idx=*/n_embs, /*end idx=*/2*n_embs).contiguous();
            auto v = qkv.slice(/*dim=*/2, /*start idx=*/2*n_embs, /*end idx=*/3*n_embs).contiguous();

            // Reshape and transpose
            // For each batch, at each head, qkv projects from nh (num heads) dims to hs (head size = emb_dim/nh) dims.
            // In hs-dim space, cosine similarity is cvalculated between query and key to obtain weights for values.
            q = q.view({B, T, n_heads, C/n_heads}).transpose(1, 2); // B, nh, T, hs
            k = k.view({B, T, n_heads, C/n_heads}).transpose(1, 2); // B, nh, T, hs
            v = v.view({B, T, n_heads, C/n_heads}).transpose(1, 2); // B, nh, T, hs

            /* Vanilla weight calculation
            // Attention weight calculation
            auto w = torch::matmul(q, k.transpose(-2, -1)) * (1.0 / std::sqrt(k.size(-1)));
            // Same logic as bias[:, :, :T, :T] in python
            w = w.masked_fill(bias.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, T), torch::indexing::Slice(0, T)}) == 0, -INFINITY);
            w = torch::nn::functional::softmax(w, -1); // B, nh, T, T
            auto y = torch::matmul(w, v); // B, nh, T, hs
            */

            // Flash attention
            auto y = std::get<0>(at::_scaled_dot_product_attention(/*query=*/q, /*key=*/k, /*value=*/v, 
                                                                   /*attn_mask=*/{}, /*dropout_p=*/0.0,
                                                                   /*need_weights=*/true, /*is_causal=*/true));
            

            // Reassemble all head outputs side by side
            // B, nh, T, hs -> B, T, nh, hs -> B, T, C (nh*hs)
            y = y.transpose(1, 2).contiguous().view({B, T, C});
            return c_proj->forward(y);
        }

    private: 
        c10::optional<at::Tensor> _opt = torch::nullopt;

};

class MLP : public torch::nn::Module {
    // Feedforward network in the transformer
    public:
        torch::nn::Linear c_fc{nullptr};
        torch::nn::GELU gelu{nullptr};
        torch::nn::Linear c_proj{nullptr};

        MLP(int embedding_dims, float dropout_prob, int seed_num){
            torch::manual_seed(seed_num);
            // From the original paper, the inner layer has a multiplier of 4
            c_fc = register_module("c_fc", torch::nn::Linear(embedding_dims, 4*embedding_dims));
            gelu = register_module("gelu", torch::nn::GELU(torch::nn::GELUOptions().approximate("tanh")));
            c_proj = register_module("c_proj", torch::nn::Linear(4*embedding_dims, embedding_dims));
        }

        torch::Tensor forward(torch::Tensor x){
            x = c_fc->forward(x);
            x = gelu->forward(x);
            return c_proj->forward(x);
        }
};

class AttentionBlock : public torch::nn::Module {
    // Transformer's attention block - communication followed by computation (multihead attention then feedforward)
    public:
        std::shared_ptr<CausalSelfAttention> c_attn{nullptr};
        torch::nn::LayerNorm layer_norm1{nullptr};
        std::shared_ptr<MLP> mlp{nullptr};
        torch::nn::LayerNorm layer_norm2{nullptr};

        AttentionBlock(int context_window_size, int embedding_dims, int n_heads, float dropout_prob, int seed_num){
            torch::manual_seed(seed_num);
            c_attn = register_module("causal_self_attention", std::make_shared<CausalSelfAttention>(context_window_size, embedding_dims, n_heads, dropout_prob, seed_num));
            layer_norm1 = register_module("layer_norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
            mlp = register_module("mlp", std::make_shared<MLP>(embedding_dims, dropout_prob, seed_num));
            layer_norm2 = register_module("layer_norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
        }

        torch::Tensor forward(torch::Tensor x){
            // Layer norms are applied prior to attention and feedforward.
            auto y1 = layer_norm1->forward(x);
            y1 = x + c_attn->forward(y1);

            auto y2 = layer_norm2->forward(y1);
            y2 = y1 + mlp->forward(y2);

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

        torch::Tensor forward(torch::Tensor x){
            // x shape B, T(token length)
            auto sizes = x.sizes();
            int B = sizes[0];
            int T = sizes[1];

            auto positions = torch::arange(0, T, torch::kLong).to(x.device()); // T
            auto pos_embeddings = position_embedding->forward(positions); // T, embedding_dims
            auto tok_embeddings = token_embedding->forward(x); // B, T, embedding_dims
            x = tok_embeddings + pos_embeddings;

            // Attention blocks
            x = attn_blocks->forward(x); // B, T, embedding_dims

            // Final layer norm
            return layer_norm->forward(x);
        }
};