from transformers import AutoTokenizer, GPTJForCausalLM, pipeline
import torch

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    # revision="float16",
    # torch_dtype=torch.float16,
)

torch.save(model, "gpt-j-6B.pt")