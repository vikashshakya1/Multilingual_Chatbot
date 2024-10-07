# preprocessing/datapreprocessing.py
import os

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def preprocess_all_datasets():
    english_data = load_data(os.path.join('data', 'data\banking_articles_English.txt'))
    hindi_data = load_data(os.path.join('data', 'data\Banking_Articles_Hindi.txt'))
    malayalam_data = load_data(os.path.join('data', 'data\banking_articles_Malayalam.txt'))
    
    return english_data, hindi_data, malayalam_data
