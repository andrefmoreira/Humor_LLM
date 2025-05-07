
import os
import gc
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import T5Tokenizer

# Directories for saving datasets
dataset_dir = './processed_datasets'
os.makedirs(dataset_dir, exist_ok=True)

# Load the dataset
dataset = load_dataset("Superar/Puntuguese")

# Combine all splits into a single dataset
full_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

# Group dataset into pairs of normal and humorous text by ID
pairs = {}
for i in full_dataset:
    id = i['id'].split('.')[0] + '.' + i['id'].split('.')[1]
    label = 'NH' if i['label'] == 0 else 'H'
    text = i['text']
    if id not in pairs:
        pairs[id] = {}
    pairs[id][label] = text

# Transform into a list of dictionaries
data = [{'id': joke_id, 'original': content['NH'], 'transformed': content['H']} for joke_id, content in pairs.items()]

# Create Hugging Face dataset from dictionary
dataset = Dataset.from_dict({key: [item[key] for item in data] for key in data[0]})

# Split dataset into train, validation, and test sets
train_temp_split = dataset.train_test_split(test_size=0.4)
test_val_split = train_temp_split['test'].train_test_split(test_size=0.5)

# Final splits
train_dataset = train_temp_split['train']
validation_dataset = test_val_split['train']
test_dataset = test_val_split['test']

# Save each split as files for later use
train_dataset.save_to_disk(os.path.join(dataset_dir, "train"))
validation_dataset.save_to_disk(os.path.join(dataset_dir, "validation"))
test_dataset.save_to_disk(os.path.join(dataset_dir, "test"))

# Define file path
train_output_file = "train_data.txt"

# Open the file and write each pair
with open(train_output_file, "w", encoding="utf-8") as f:
    for example in train_dataset:
        f.write(f"Original: {example['original']}\n")
        f.write(f"Transformed: {example['transformed']}\n")
        f.write("="*80 + "\n")  # Separator for readability

print(f"Train input and output saved to {train_output_file}")
print(f"Datasets saved to {dataset_dir}")
