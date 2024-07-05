#include "gpt2.h"
#include "utils.h"

void GPT_trainer(const std::string& data_path, const std::string& tiktoken_conf, const std::string& gpt_model){

    // __________________________________________________________________________________________________________
    // Initialization
    // 1. Instantiate tiktokenizer
    // 2. Identify the device type to work with
    // 3. Create gpt config based on selected gpt_model
    // __________________________________________________________________________________________________________
    tokenizer::tiktoken tokenizer(tiktoken_conf);

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Device run_device(device_type);

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

    // __________________________________________________________________________________________________________
    // Data parsing, input data tensor creation
    // __________________________________________________________________________________________________________
    std::cout<<"[INFO]  Data parsing....."<<std::endl;
    torch::Tensor tokens = preprocessing::data_parser("../data/input.txt", tokenizer, run_device);
    std::cout<<"[INFO]  Data parsing completed."<<std::endl;

    // Split into train/val tensors.
    std::cout<<"[INFO]  Splitting data into train/val set....."<<std::endl;
    torch::Tensor train_data, val_data;
    preprocessing::split_dataset(0.9, tokens, train_data, val_data);
    std::cout<<"[INFO]  Train/val set gathered."<<std::endl;


}