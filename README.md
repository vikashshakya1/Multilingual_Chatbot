# Multilingual Chatbot

This multilingual chatbot is capable of conversing in Hindi, English, and Malayalam, with separate models trained for each language.

## Directory Structure
multilingual-chatbot/
│
├── configs/                # Configuration files
│   └── config.json         # Example configuration file
│
├── Data/                   # Input text files
│   ├── banking_articles_English.txt
│   ├── banking_articles_Hindi.txt
│   └── banking_articles_Malayalam.txt
│
├── Models/                 # Model training scripts
│   ├── train_english.py
│   ├── train_hindi.py
│   └── train_malayalam.py
│
├── preprocessing/          # Preprocessing scripts
│   ├── datapreprocessing.py
│   └── tokenizer.py
│
├── chatbot.py              # Chatbot interface
│
└── README.md               # Project documentation

