# gpt2
GPT2-124m implementation in C++

1. Transformer architecture implemented with LibTorch
2. Flash transformer for faster calculation and data fetch cycle
3. MPI for multi-process data loading and model gradient updates (only tested with CPU, DDP seems to be supported only on PyTorch, not LibTorch yet)
4. Autocast to TF32/BF16 types (NVIDIA Ampere architecture or above required), or FP16 type for faster training.
5. Function to load pre-trained weights (from custom trained model, and HuggingFace GPT-2 model)

This implementation is created with LibTorch 1.13.0 with CUDA 12.0
