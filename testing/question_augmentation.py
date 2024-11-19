import json

import nltk
from textaugment import EDA

from src.dataloaders import ConfigLoad

nltk.download("wordnet")
nltk.download("stopwords")

config_loader = ConfigLoad()
config = config_loader.config

path = config["data"]["qa_file_path"]

with open(path, "r") as file:
    data = json.load(file)

eda = EDA()

for item in data:
    question = item["question"]
    synonym_replacement = eda.synonym_replacement(question)
    item["synonym_replacement"] = synonym_replacement
    random_insertion = eda.random_insertion(question)
    item["random_insertion"] = random_insertion
    random_swap = eda.random_swap(question)
    item["random_swap"] = random_swap
    random_deletion = eda.random_deletion(question)
    item["random_deletion"] = random_deletion

    random_complex = eda.random_insertion(
        eda.random_swap(eda.synonym_replacement(eda.random_deletion(question)))
    )
    item["random_complex"] = random_complex


with open("eval-augmentation-new.json", "w") as file:
    updated_json = json.dump(data, file, indent=4)
