import torch
from transformers import GPT2LMHeadModel
from torch import nn

class GPT2Wrapper(nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model
    
    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs.logits  # Return only the logits

def save_gpt2_weights(model_type='gpt2'):
    # Load the model from Hugging Face
    model = GPT2LMHeadModel.from_pretrained(model_type)
    wrapped_model = GPT2Wrapper(model)
    
    # Create a dummy input tensor with the appropriate shape
    example_input = torch.randint(0, model.config.vocab_size, (1, model.config.n_positions), dtype=torch.long)
    
    # Trace the wrapper model with the example input
    traced_script_module = torch.jit.trace(wrapped_model, example_input)
    traced_script_module.save(f"{model_type}_weights.pt")

save_gpt2_weights('gpt2')