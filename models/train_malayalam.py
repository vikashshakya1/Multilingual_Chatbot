# models/train_malayalam.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from preprocessing.datapreprocessing import preprocess_data

# Load the tokenizer and model for Malayalam
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load and preprocess the Malayalam data
malayalam_data = preprocess_data("Malayalam")
data = malayalam_data

# Tokenize the data
def tokenize_data(data):
    return tokenizer(data, padding=True, truncation=True, return_tensors="pt", max_length=128)

tokenized_data = tokenize_data(data)

# Create dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

dataset = CustomDataset(tokenized_data)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results/malayalam',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()
trainer.save_model("models/malayalam_chatbot_model")
