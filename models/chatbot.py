# models/chatbot.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the models and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
english_model = GPT2LMHeadModel.from_pretrained("models/train_english.py")
hindi_model = GPT2LMHeadModel.from_pretrained("models/train_hindi.py")
malayalam_model = GPT2LMHeadModel.from_pretrained("models/train_malayalam.py")

def generate_response(model, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Simple chatbot loop
if __name__ == "__main__":
    print("Multilingual Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Simple language detection (you can replace this with a proper language detection mechanism)
        if any(char.isascii() for char in user_input):  # English detection
            response = generate_response(english_model, user_input)
        elif any('\u0900' <= char <= '\u097F' for char in user_input):  # Hindi detection
            response = generate_response(hindi_model, user_input)
        elif any('\u0D00' <= char <= '\u0D7F' for char in user_input):  # Malayalam detection
            response = generate_response(malayalam_model, user_input)
        else:
            response = "Sorry, I don't understand that language."

        print(f"Bot: {response}")
