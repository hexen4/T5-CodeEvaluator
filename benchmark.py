from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

device:str = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
model.eval()
for prompt in ["create a python script that checks prime numbers up to n"]:
    print("Input:", prompt)
    inputTokens = tokenizer("create a python script based on the following prompt: {}".format(prompt), return_tensors="pt").to(device)
    outputs = model.generate(inputTokens['input_ids'], attention_mask=inputTokens['attention_mask'], max_new_tokens=50)
    print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))