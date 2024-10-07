# Multilingual Chatbot

This multilingual chatbot is capable of conversing in Hindi, English, and Malayalam, with separate models trained for each language.

## Directory Structure
**multilingual_chatbot/**
├── configs/
│   └── config.py
├── data/
│   ├── banking_articles_English.txt
│   ├── banking_articles_Hindi.txt
│   └── banking_articles_Malayalam.txt
├── models/
│   ├── train_english.py
│   ├── train_hindi.py
│   └── train_malayalam.py
├── preprocessing/
│   ├── datapreprocessing.py
│   └── tokenizer.py
├── requirements.txt
└── README.md
