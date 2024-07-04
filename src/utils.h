#pragma once
#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <regex>
 
namespace utils{
    // Extract the layer index from "h.layer_idx" form in hugging face weight pt file.
    int extract_layer_num(const std::string& input_string) {
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
    bool ends_with(const std::string& fullString, const std::string& ending) {
        if (fullString.length() >= ending.length()) {
            return (fullString.substr(fullString.length() - ending.length()) == ending);
        } else {
            return false;
        }
    }
}

// Load the pretrained GPT model weights (ideally downloaded from HuggingFace)
inline void load_from_pretrained(GPT& model, const std::string& weights_path){
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
