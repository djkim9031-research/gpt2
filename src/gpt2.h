#include <torch/torch.h>
#include <cmath>
#include <memory>
#include <unordered_map>

#include "transformer.h"

struct GPTConfig{
    int context_win_size;
    int vocab_size;
    int n_layers;
    int n_heads;
    int n_embs;

    GPTConfig(int context_win_size = 256, int vocab_size = 65, int n_layers = 6, int n_heads = 6, int n_embs = 384)
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
            return transformer->forward(x);
        }
};