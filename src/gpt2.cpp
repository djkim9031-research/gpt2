#include "gpt2.h"
#include "utils.h"

std::vector<std::vector<uint64_t>> GPT::inference(const std::vector<uint64_t>& token_encodings, 
                                                  const int num_return_sequences, 
                                                  const int max_generation_length, 
                                                  const torch::Device& device){

    this->eval(); // turn the model to eval mode.

    // Convert token encodings to a tensor and replicate it by num_return_sequences
    std::vector<int64_t> token_encodings_int64(token_encodings.begin(), token_encodings.end());
    // Token shape will be (num_return_sequences, token_encodings.size()) = (B, T)
    torch::Tensor tokens = torch::tensor(token_encodings_int64, torch::kLong).unsqueeze(0).repeat({num_return_sequences, 1});
    tokens = tokens.to(device);

    set_seed(42);

    // Generate tokens
    while(tokens.size(1) < max_generation_length){
        // Forward pass to get logits
        torch::Tensor logits;
        {
            torch::NoGradGuard no_grad;
            // B, T, C
            logits = this->forward(tokens);
        }
        
        // Only interested in the logit at the last position
        // B, 1, C
        logits = logits.slice(/*dim=*/1, /*start idx=*/logits.size(1)-1, /*end idx=*/logits.size(1));

        // Get the probabilites of vocabs
        torch::Tensor probs = torch::softmax(logits, -1);
        
        // Perform top-k sampling
        int k = 50;
        auto top_k = probs.topk(k, -1);
        torch::Tensor top_k_probs = std::get<0>(top_k);
        torch::Tensor top_k_indices = std::get<1>(top_k);
        // Squeeze the 2nd dimension (B, 1, C) => (B, C)
        top_k_probs = top_k_probs.squeeze(1);
        top_k_indices = top_k_indices.squeeze(1);

        // Select a token from the top-k probabilities.
        // (B, 1), 1 selected from top-k.
        // Multinomial picks will generate indices of top_k_probs tensors, which are ordered by high -> low probs.
        // To pick the corresponding vocab index, top_k_indices should be queried with idx key (arranged probs).
        torch::Tensor idx = torch::multinomial(top_k_probs, /*num samples=*/1, /*replacement=*/true);
        
        // Gather the corresponding vocab indices.
        torch::Tensor xcols = top_k_indices.gather(/*dim=*/1, /*index=*/idx);

        // Append to the sequence
        // (B, T) => (B, T + 1)
        tokens = torch::cat({tokens, xcols}, /*dim=*/1);
    }


    tokens = tokens.cpu();
    std::vector<std::vector<uint64_t>> generated_tokens(num_return_sequences, std::vector<uint64_t>(max_generation_length));
    for(size_t i=0; i<num_return_sequences; ++i){
        for(size_t j=0; j<max_generation_length; ++j){
            generated_tokens[i][j] = static_cast<uint64_t>(tokens[i][j].item<int64_t>());
        }
    }
    return generated_tokens;
}