import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset
import wandb

moralizing = json.load(open('moralizing.json'))
# shuffle the moralizing examples
np.random.shuffle(moralizing)
N_SAMPLES = len(moralizing)

texts = [text.lower() for text in moralizing[:N_SAMPLES]]
labels = [1] * N_SAMPLES
regular = json.load(open('regular.json'))[:N_SAMPLES]
np.random.shuffle(regular)
texts += [text.lower() for text in regular[:N_SAMPLES]]
labels += [0] * N_SAMPLES

# Free up memory
del moralizing, regular

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize wandb
wandb.init(project="moralizing_classifier", entity="yourusername")

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize the input (transform text to tokens)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=192)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=192)

# Custom dataset class
class MoralizingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Convert to Dataset
train_dataset = MoralizingDataset(train_encodings, train_labels)
val_dataset = MoralizingDataset(val_encodings, val_labels)


# Adjustments to TrainingArguments to include a custom optimizer and scheduler
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=8,   # Batch size per device during training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=50,
    evaluation_strategy="steps",     # Evaluate the model every `logging_steps` steps
    save_strategy="steps",
    save_steps=100,                  # Save checkpoint every `save_steps` steps
    load_best_model_at_end=True,     # Load the best model at the end of training
    metric_for_best_model="loss",    # Use loss to evaluate the best model
    greater_is_better=False,
    max_grad_norm=1.0,               # Gradient clipping
)

# Initialize Trainer with custom optimizer and scheduler
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# Train the model
trainer.train()
