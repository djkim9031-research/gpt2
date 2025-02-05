#include "gpt2.h"
#include "utils.h"
#include <cassert>

#include <chrono>
#include <ATen/autocast_mode.h>

torch::optim::Optimizer* GPT::configure_optimizers(const float weight_decay,
                                                   const float learning_rate,
                                                   const float beta1,
                                                   const float beta2,
                                                   const float eps){
    // Get all trainable parameters.
    auto params = this->parameters();

    // Separate parameters into groups based on dimensionality
    // Any params with 2D or above will be weight decayed, otherwise they won't.
    // i.e., all weight tensors with matmults + embeddings will decay, while biases and layernorms won't
    std::vector<torch::Tensor> decay_params;
    std::vector<torch::Tensor> nodecay_params;
    for (const auto& param : params) {
        if (param.requires_grad()) {
            if (param.dim() >= 2) {
                decay_params.push_back(param);
            } else {
                nodecay_params.push_back(param);
            }
        }
    }

    // Print information about parameter groups
    int num_decay_params = 0;
    for (const auto& p : decay_params) {
        num_decay_params += p.numel();
    }
    int num_nodecay_params = 0;
    for (const auto& p : nodecay_params) {
        num_nodecay_params += p.numel();
    }
    std::cout << "[INFO]  Number of decayed parameter tensors: " << decay_params.size()
              << ", with " << num_decay_params << " parameters\n";
    std::cout << "[INFO]  Number of non-decayed parameter tensors: " << nodecay_params.size()
              << ", with " << num_nodecay_params << " parameters\n";

    // Create optimizer groups
    std::vector<torch::optim::OptimizerParamGroup> optim_groups;

    // Weight decay group
    std::unique_ptr<torch::optim::AdamWOptions> decay_options = std::make_unique<torch::optim::AdamWOptions>();
    decay_options->weight_decay(weight_decay);
    torch::optim::OptimizerParamGroup decay_group(decay_params, std::move(decay_options));
    optim_groups.push_back(decay_group);

    // No decay group
    std::unique_ptr<torch::optim::AdamWOptions> nodecay_options = std::make_unique<torch::optim::AdamWOptions>();
    nodecay_options->weight_decay(0.0);
    torch::optim::OptimizerParamGroup nodecay_group(nodecay_params, std::move(nodecay_options));
    optim_groups.push_back(nodecay_group);

    // Create optimizer with customized optim groups
    torch::optim::Optimizer* optimizer = new torch::optim::AdamW(optim_groups, torch::optim::AdamWOptions(learning_rate).betas({beta1, beta2}).eps(eps));
    return optimizer;
}

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
    // 1. DDP (Distributed Data Parallel) pipeline setup.
    // 2. Instantiate tiktokenizer
    // 3. Identify the device type to work with
    // 4. Create gpt config based on selected gpt_model
    // __________________________________________________________________________________________________________
    // DDP setup
    int process_rank, world_size;
    bool is_master = trainer::ddp_pipeline_setup(process_rank, world_size);

    // Tokenizer setup
    tokenizer::tiktoken tokenizer(tiktoken_conf);

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Device run_device(device_type);
    std::cout<<"[INFO]  Process "<<process_rank<<" running on "<<device_type<<std::endl;

    std::unique_ptr<GPTConfig> config{nullptr};

    if(gpt_model == "gpt2"){
        // Original gpt2's context_win_size = 50257, but making it 2's exp help CUDA kernel resource allocation.
        config = std::make_unique<GPTConfig>(1024, 50304, 12, 12, 768); // 124M+ params
        std::cout<<"[INFO]  Process "<<process_rank<<" GPT2 model config generated."<<std::endl;
    } else if (gpt_model == "gpt2-medium"){
        config = std::make_unique<GPTConfig>(1024, 50304, 24, 16, 1024); // 350M+ params
        std::cout<<"[INFO]  Process "<<process_rank<<" GPT2-medium model config generated."<<std::endl;
    } else if (gpt_model == "gpt2-large"){
        config = std::make_unique<GPTConfig>(1024, 50304, 36, 20, 1280); // 774M+ params
        std::cout<<"[INFO]  Process "<<process_rank<<" GPT2-large model config generated."<<std::endl;
    } else if (gpt_model == "gpt2-xl"){
        config = std::make_unique<GPTConfig>(1024, 50304, 48, 25, 1600); // 1.558B+ params
        std::cout<<"[INFO]  Process "<<process_rank<<" GPT2-xl model config generated."<<std::endl;
    } else{
        throw std::invalid_argument(gpt_model+" does not exist. Try one of [gpt2, gpt2-medium, gpt2-large, gpt2-xl]");
    }

    // __________________________________________________________________________________________________________
    // Set hyperparameters.
    // __________________________________________________________________________________________________________
    utils::set_seed(42);
    // Training params.
    int total_batch_size = 524288; // 2^19, ~0.5M in number of tokens (from GPT paper)
    int batch_size = 1;
    int step = 0;
    int max_steps = 50;

    // Learning rate decay params.
    float max_lr = 6e-4;
    float min_lr = max_lr*0.1;
    int warmup_steps = 10;

    // Adam optimizer params.
    float beta1 = 0.9;
    float beta2 = 0.95;
    float eps = 1e-8;
    float weight_decay = 0.1;

    // Flop calculation, gradient accumulation params
    int num_tokens = batch_size * config->context_win_size;
    assert(total_batch_size%(num_tokens*world_size) == 0);
    int grad_accum_steps = total_batch_size/(num_tokens*world_size);
    // __________________________________________________________________________________________________________
    // Data parsing, input data tensor creation
    // __________________________________________________________________________________________________________
    std::cout<<"[INFO]  Process "<<process_rank<<" Data parsing....."<<std::endl;
    torch::Tensor tokens = preprocessing::data_parser("../data/input.txt", tokenizer, process_rank, world_size, run_device);
    std::cout<<"[INFO]  Process "<<process_rank<<" Data parsing completed."<<std::endl;

    // Split into train/val tensors.
    std::cout<<"[INFO]  Process "<<process_rank<<" Splitting data into train/val set....."<<std::endl;
    torch::Tensor train_data, val_data;
    preprocessing::split_dataset(0.9, tokens, train_data, val_data);
    std::cout<<"[INFO]  Process "<<process_rank<<" Train/val set gathered."<<std::endl;

    // __________________________________________________________________________________________________________
    // GPT model construction.
    // __________________________________________________________________________________________________________

    GPT model(*config);
    model.to(run_device);
    model.train();
    std::cout<<"[INFO]  Process "<<process_rank<<" GPT2 model created."<<std::endl;

    // Learning rate scheduler
    trainer::lr_scheduler lr_scheduler(max_lr, min_lr, warmup_steps, max_steps);

    // Optimizer
    //torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(max_lr).betas({beta1, beta2}).eps(eps).weight_decay(weight_decay_factor));
    auto optimizer = model.configure_optimizers(weight_decay, max_lr, beta1, beta2, eps);
    std::cout<<"[INFO]  Process "<<process_rank<<" Total desired batch size: "<<total_batch_size<<" => Caculated gradient accumulation steps: "<<grad_accum_steps<<std::endl;   
    // __________________________________________________________________________________________________________
    // Training loop
    // => Make sure to set a reasonalbe batch_size. At least 8GB VRAM on CUDA device seems necessary.
    // __________________________________________________________________________________________________________
    std::cout<<"[INFO]  Process "<<process_rank<<" Training started....."<<std::endl;
    std::cout<<"_________________________________________________________________________________________"<<std::endl;
    while(step < max_steps){
        torch::Tensor accumulated_loss = torch::zeros({1}, torch::kFloat).to(run_device);
        auto t0 = std::chrono::high_resolution_clock::now();

        optimizer->zero_grad();

        for(int grad_step=0; grad_step<grad_accum_steps; ++grad_step){
            // Create batch
            torch::Tensor x_train, y_train;
            preprocessing::create_batch(batch_size, config->context_win_size, train_data, x_train, y_train);

            // Forward propagation, with automatic mixed precision for speed gain.
            // If the CUDA architecture supports FP16, this should speed up with half-precision.
            // If you have Ampere architecture, BF16 (16bit) w/o gradient scaling necessity,
            // TF32 (19bits) are supported but I haven't tested to see if libTorch supports it.
            // Use the at::autocast with caution if the architecture only support FP16, in which case gradeint scaling is necessary,
            // especially if there are known vanishing gradient issues.
            // However, LibTorch doesn't have built-in gradient scaler yet.
            // TODO: If TF32/BF16 mixed precision are supported, or gradient scaler is available for FP16,
            // {
            //  at::autocast::set_enabled(true);
            //  auto logits = model.forward(x_train);
            //  auto loss = torch::nn::functional::cross_entropy(logits.view({-1, logits.size(2)}), y_train.view({-1}));
            //  at::autocast::clear_cache();
            //  at::autocast::set_enabled(false);
            //  loss /= float(grad_accum_steps);
            //}
            auto logits = model.forward(x_train);

            // logits [B, T, C], y_train [B, T]
            // loss is calculated over C dim, and B,T dims are combined.
            auto loss = torch::nn::functional::cross_entropy(logits.view({-1, logits.size(2)}), y_train.view({-1}));
            loss /= float(grad_accum_steps);
            accumulated_loss += loss.detach();

            // Backward propagation
            loss.backward();
        }
        // float value of avg_loss (per process)
        float avg_loss = accumulated_loss.item<float>();
        float local_avg_loss = accumulated_loss.item<float>();

        // global grad norm clipping at 1.0
        torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);

        // Interprocess communications - average out gradients in each model copy (per process)
        // Currently only tested MPI comm on CPU device. TODO: comm on CUDA.
        if(world_size > 1){
            auto local_grads = trainer::collect_local_gradients(model);
            local_grads = local_grads.cpu();
            torch::Tensor global_grads = torch::zeros_like(local_grads);
            MPI_Allreduce(local_grads.data_ptr<float>(), global_grads.data_ptr<float>(), local_grads.numel(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            avg_loss = 0.0;
            MPI_Allreduce(&local_avg_loss, &avg_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            global_grads /= float(world_size);
            avg_loss /= float(world_size);
            trainer::update_gradients(model, global_grads);
            if(is_master){
                std::cout<<"[INFO]  Gradients synch between all processed done."<<std::endl;
            }
        }
        

        // Learning rate update
        float curr_lr = lr_scheduler.get_lr(step);
        for (auto param_group : optimizer->param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(param_group.options()).lr(curr_lr);
        }
        optimizer->step();

        if(torch::cuda::is_available()){
            torch::cuda::synchronize();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = t1 - t0;
        float duration = elapsed.count();

        if(is_master){
            std::cout<<"[INFO]  Step "<<step<<", lr = "<<curr_lr<<", loss = "<<avg_loss<<", Elapsed = "<<duration<<
                       "(ms), Processed tokens = "<<float(num_tokens*grad_accum_steps*world_size)/(duration/1000)<<"(tok/sec)"<<std::endl;
        }    
        step++;
    }
    if(is_master){
        std::cout<<"_________________________________________________________________________________________"<<std::endl;
        std::cout<<"[INFO]  Training completed."<<std::endl;
    }
    
    trainer::ddp_pipeline_cleanup();
}