from transformers import AutoTokenizer
import torch

device = torch.device("cpu")

model = torch.load("gpt-j-6B.pt")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

while True:
    prompt = input("\nInput: ")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=20, pad_token_id=50256)
    generated_text = tokenizer.decode(generated_ids[0])
    print("\nOutput: " + generated_text)