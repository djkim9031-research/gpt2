#include "gpt2.h"
#include "utils.h"

void GPT::_init_weights(torch::nn::Module& curr_module, const std::string& curr_module_name){
    if(auto linear = dynamic_cast<torch::nn::LinearImpl*>(&curr_module)){
            float std = 0.02;
            if(utils::ends_with(curr_module_name, "c_proj")){
                std *= 1/std::sqrt(2*(this->config.n_layers));
            }
            // zero mean, 0.02 std for linear layers, but for proj layers, normalize it with
            // multipler 1/sqrt(2*n_layers) to mitigate the expansion of std from residual connections.
            torch::nn::init::normal_(linear->weight, 0.0, std);
            if(linear->bias.defined()){
                torch::nn::init::zeros_(linear->bias);
            }
        } else if (auto embedding = dynamic_cast<torch::nn::EmbeddingImpl*>(&curr_module)){
            // zero mean, 0.02 std as GPT2 is implemented.
            torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
        }
}

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
    std::cout<<"[INFO]  Running on "<<device_type<<std::endl;

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

    // Set hyperparameters
    utils::set_seed(42);
    int batch_size = 1;
    int step = 1;
    int max_steps = 10;
    float learning_rate = 3e-4;

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

    // __________________________________________________________________________________________________________
    // GPT model construction.
    // __________________________________________________________________________________________________________
    GPT model(*config);
    model.to(run_device);
    model.train();

    // Optimizer
    torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(learning_rate));

    // __________________________________________________________________________________________________________
    // Training loop
    // => Make sure to set a reasonalbe batch_size. At least 8GB VRAM on CUDA device seems necessary.
    // __________________________________________________________________________________________________________
    std::cout<<"[INFO]  Training started....."<<std::endl;
    std::cout<<"_________________________________________________________________________________________"<<std::endl;
    while(step < max_steps){
        // Create batch
        torch::Tensor x_train, y_train;
        preprocessing::create_batch(batch_size, config->context_win_size, train_data, x_train, y_train);

        // Forward propagation
        auto logits = model.forward(x_train);

        // logits [B, T, C], y_train [B, T]
        // loss is calculated over C dim, and B,T dims are combined.
        auto loss = torch::nn::functional::cross_entropy(logits.view({-1, logits.size(2)}), y_train.view({-1}));

        // Backward propagation
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout<<"[INFO]  Step "<<step<<", loss = "<<loss.item<float>()<<std::endl;
        step++;
    }
    std::cout<<"_________________________________________________________________________________________"<<std::endl;
    std::cout<<"[INFO]  Training completed."<<std::endl;

}