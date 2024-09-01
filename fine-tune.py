import torch
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Conv1D
import transformers
class CodeEvaluator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules= ["T5LayerCrossAttention"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        #print(self.model)
        #self.model = prepare_model_for_kbit_training(self.model)
        #self.model = get_peft_model(self.model, self.config)
        self.tiny_codes = None
        self.codeparrot = None
        #self.load_datasets()
    def load_datasets(self):
        if self.tiny_codes is None:
            self.tiny_codes = load_dataset('tiny-codes')
            print("Loaded tiny-codes dataset.")
        else:
            print("tiny-codes dataset already loaded.")  # This will check and load if not already loaded
        if self.codeparrot is None:
            self.codeparrot = load_dataset('codeparrot-clean')
            print("Loaded codeparrot-clean dataset.")
        else:
            print("codeparrot-clean dataset already loaded.")
        return self.codeparrot,self.tiny_codes

    def tokenize_function(self,data):
        # extract text
        text = data["example"] #set this to the context or smth

        #tokenize and truncate text
        self.tokenizer.truncation_side = "left"
        tokenized_inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512
        )

        return tokenized_inputs
    
    def train(self):
        # Perform training here
        self.model.print_trainable_parameters()
        tokenized_data_codeparrot = self.codeparrot.map(self.tokenize_function, batched=True)
        tokenized_data_codeparrot = self.tiny_codes.map(self.tokenize_function, batched=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        # hyperparameters
        lr = 2e-4
        batch_size = 4
        num_epochs = 10

        # define training arguments
        training_args = transformers.TrainingArguments(
            output_dir= "./results",
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
            model=self.model,
            train_dataset=self.tokenized_data["train"],
            eval_dataset=self.tokenized_data["test"],
            args=training_args,
            data_collator=data_collator
        )

        # train model
        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

        # renable warnings
        self.model.config.use_cache = True
    def evaluate(self):
        # Perform evaluation here
        pass

    def get_specific_layer_names(self):
        # Create a list to store the layer names
        layer_names = []
        
        # Recursively visit all modules and submodules
        for name, module in self.model.named_modules():
            # Check if the module is an instance of the specified layers
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                # model name parsing 

                layer_names.append(name)
        
        return layer_names


if __name__ == "__main__":
    codeval = CodeEvaluator()  # This will load datasets during initialization
    layers = codeval.get_specific_layer_names()
    print(list(set(layers)))
    codeparrot,tiny_codes = codeval.load_datasets()  # Access tiny-codes dataset
    codeval.train()  # Train the model