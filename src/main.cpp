#include <torch/torch.h>
#include <vector>

#include <unistd.h>
#include <iostream>
#include <string>
#include "gpt2.h"

int main(int argc, char **argv) {

    int opt = 0;
    std::string mode;
    std::string tiktoken_conf;
    std::string input_string;
    int num_inference_variants;
    int num_target_tokens;

    while ((opt = getopt(argc, argv, "m:t:i:n:l:")) != -1) {
        switch (opt) {
        case 'm':
            mode = optarg;
        case 't':
            tiktoken_conf = optarg;
            break;
        case 'i':
            input_string = optarg;
            break;
        case 'n':
            num_inference_variants = std::stoi(optarg);
            break;
        case 'l':
            num_target_tokens = std::stoi(optarg);
            break;
        default:
            std::cerr << "unknown command option" << std::endl;
            return -1;
            break;
        }
    }
    

    if(mode == "train"){
        std::cout<<"Training logic triggered"<<std::endl;
    } else if(mode == "test"){
        GPT_playground(tiktoken_conf, input_string, num_target_tokens, num_inference_variants);
    }

    return 0;
}

