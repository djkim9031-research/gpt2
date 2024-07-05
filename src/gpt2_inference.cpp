#include "gpt2.h"
#include "utils.h"

std::vector<std::vector<uint64_t>> GPT::inference(const std::vector<int64_t>& token_encodings, 
                                                  const int num_return_sequences, 
                                                  const int max_generation_length, 
                                                  const torch::Device& device){

    this->eval(); // turn the model to eval mode.

    // Token shape will be (num_return_sequences, token_encodings.size()) = (B, T)
    torch::Tensor tokens = torch::tensor(token_encodings, torch::kLong).unsqueeze(0).repeat({num_return_sequences, 1});
    tokens = tokens.to(device);

    utils::set_seed(42);

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


void GPT_playground(const std::string& input_string, 
                    const std::string& tiktoken_conf, 
                    const int target_sequence_length, 
                    const int num_output_variants,
                    const std::string& gpt_model){

    // Instantiate tiktoken tokenizer.
    tokenizer::tiktoken tokenizer(tiktoken_conf);
    
    // Encode the input language string to tiktoken encodings.
    std::vector<int64_t> tokens = tokenizer.encode(input_string);

    // Identify if GPU is available.
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Device run_device(device_type);

    // Construct the GPT2 model.
    std::unique_ptr<GPTConfig> config{nullptr};

    if(gpt_model == "gpt2"){
        config = std::make_unique<GPTConfig>(1024, 50257, 12, 12, 768); // 124M params
        std::cout<<"[INFO]  GPT2 model config generated."<<std::endl;
    } else if (gpt_model == "gpt2-medium"){
        config = std::make_unique<GPTConfig>(1024, 50257, 24, 16, 1024); // 350M params
        std::cout<<"[INFO]  GPT2-medium model config generated."<<std::endl;
    } else if (gpt_model == "gpt2-large"){
        config = std::make_unique<GPTConfig>(1024, 50257, 36, 20, 1280); // 774M params
        std::cout<<"[INFO]  GPT2-large model config generated."<<std::endl;
    } else if (gpt_model == "gpt2-xl"){
        config = std::make_unique<GPTConfig>(1024, 50257, 48, 25, 1600); // 1.558B params
        std::cout<<"[INFO]  GPT2-xl model config generated."<<std::endl;
    } else{
        throw std::invalid_argument(gpt_model+" does not exist. Try one of [gpt2, gpt2-medium, gpt2-large, gpt2-xl]");
    }
    
    GPT model(*config);
    model.to(run_device);

    // Load the pretrained weights.
    // I will update this logic to load weights that are custom trained.
    pretrained::load_from_pretrained_GPT2_HF(model, "../data/gpt2_weights.pt");

    // GPT2 model inference
    std::vector<std::vector<uint64_t>> generated_tokens = model.inference(tokens, num_output_variants, target_sequence_length, run_device);

    // Decode the generated tokens, and print the outputs.
    std::cout<<"_________________________________________________________________________________________"<<std::endl;
    for(int i=0; i<num_output_variants; ++i){
        std::cout<<"[GENERATED] "<<tokenizer.decode(generated_tokens[i])<<std::endl;
        std::cout<<"_________________________________________________________________________________________"<<std::endl;
    }

    return;
}