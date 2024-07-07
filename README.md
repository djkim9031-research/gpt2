# gpt2
GPT2-124m implementation in C++

1. Transformer architecture implemented with LibTorch
2. Flash transformer for faster calculation and data fetch cycle
3. MPI for multi-process data loading and model gradient updates (only tested with CPU)
4. Autocast to TF32/BF16 types (NVIDIA Ampere architecture or above required), or FP16 type for faster training.
