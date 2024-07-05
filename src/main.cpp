#include <torch/torch.h>
#include <vector>

#include <unistd.h>
#include <iostream>
#include <string>
#include "gpt2.h"
#include "utils.h"

int main(int argc, char **argv) {

    int opt = 0;
    std::string mode;
    std::string tiktoken_conf;
    std::string input_string;
    int num_inference_variants;
    int num_target_tokens;
    std::string gpt_model;

    while ((opt = getopt(argc, argv, "m:t:i:n:l:v:")) != -1) {
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
        case 'v':
            gpt_model = optarg;
            break;
        default:
            std::cerr << "unknown command option" << std::endl;
            return -1;
            break;
        }
    }
    

    if(mode == "train"){
        GPT_trainer("../data/input.txt", tiktoken_conf, gpt_model);

    } else if(mode == "test"){
        GPT_playground(input_string, tiktoken_conf, num_target_tokens, num_inference_variants, gpt_model);
    }

    return 0;
}

