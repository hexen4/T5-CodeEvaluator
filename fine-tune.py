from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM , pipeline
import torch
#import data
tiny_codes = load_dataset('tiny-codes')
codeparrot = load_dataset('codeparrot-clean')
device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast = True)
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
model.train() # model in training mode (dropout modules are activated)
# enable gradient check pointing
model.gradient_checkpointing_enable()
# enable quantized training
model = prepare_model_for_kbit_training(model)
# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    #target_modules=["None"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# LoRA trainable version of model
model = get_peft_model(model, config)
# trainable parameter count
model.print_trainable_parameters()

def tokenize_function(examples):
    # extract text
    text = examples["example"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data_codeparrot = codeparrot.map(tokenize_function, batched=True)
tokenized_data_tiny_codes = tiny_codes.map(tokenize_function, batched=True)

# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "shawgpt-ft",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",

)

# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data_codeparrot["train"],
    eval_dataset=tokenized_data_codeparrot["test"],
    args=training_args,
    data_collator=data_collator
)

# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# renable warnings
model.config.use_cache = True