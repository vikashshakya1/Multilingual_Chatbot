# configs/__init__.py
import json

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

config = load_config('configs/config.json')  # Example config loading
