import os
import random
import gc

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report

# PEFT LoRA
from peft import LoraConfig, get_peft_model, TaskType

# ---------------- CONFIG ----------------
DATASET_NAME = 'Davlan/sib200'
DATASET_LANGUAGE = 'rus_Cyrl'
MODEL_CHECKPOINT = 'ai-forever/FRIDA'

MAX_LENGTH = 256
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 5e-4
NUM_TRAIN_EPOCHS = 10
SEED = 42
OUTPUT_DIR = './results_FRIDA_LoRA'

USE_FP16 = False
USE_BF16 = True

# ---------------- REPRO ----------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------- CLEAR GPU ----------------
torch.cuda.empty_cache()
gc.collect()

# ---------------- DATA ----------------
print("Loading dataset...")
train_ru = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='train')
val = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='validation')
test = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='test')

# Подмешиваем другие языки
try:
    eng_train = load_dataset(DATASET_NAME, 'eng_Latn', split='train')
    fra_train = load_dataset(DATASET_NAME, 'fra_Latn', split='train')
    combined_train = concatenate_datasets([train_ru, eng_train, fra_train])
    print("Combined multilingual train size:", len(combined_train))
except Exception as e:
    print("Can't load other language splits — using ru train only. Err:", e)
    combined_train = train_ru

# ---------------- LABEL MAPPING ----------------
unique_categories = sorted(list(set(train_ru['category'])))
num_labels = len(unique_categories)
label2id = {label: i for i, label in enumerate(unique_categories)}
id2label = {i: label for label, i in label2id.items()}
print("Unique categories:", unique_categories)
print("Num labels:", num_labels)

def map_to_label_ids(example):
    example['label'] = label2id.get(example['category'], -1)
    return example

combined_train = combined_train.map(map_to_label_ids)
val = val.map(map_to_label_ids)
test = test.map(map_to_label_ids)

# Фильтруем неизвестные категории
combined_train = combined_train.filter(lambda x: x['label'] != -1)
val = val.filter(lambda x: x['label'] != -1)
test = test.filter(lambda x: x['label'] != -1)

# Переименовываем для Trainer
combined_train = combined_train.rename_column('label', 'labels')
val = val.rename_column('label', 'labels')
test = test.rename_column('label', 'labels')

# ---------------- TOKENIZER & MODEL ----------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# ---------------- LoRA ----------------
print("Applying LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=[
        "SelfAttention.q",
        "SelfAttention.k",
        "SelfAttention.v",
        "SelfAttention.o",
        "EncDecAttention.q",
        "EncDecAttention.k",
        "EncDecAttention.v",
        "EncDecAttention.o"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------- TOKENIZATION ----------------
def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH)

tokenized_train = combined_train.map(tokenize_fn, batched=True)
tokenized_val = val.map(tokenize_fn, batched=True)
tokenized_test = test.map(tokenize_fn, batched=True)

tokenized_train.set_format(type='torch')
tokenized_val.set_format(type='torch')
tokenized_test.set_format(type='torch')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------- METRICS ----------------
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        'macro_precision': precision_score(labels, preds, average='macro', zero_division=0),
        'macro_recall': recall_score(labels, preds, average='macro', zero_division=0),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'accuracy': accuracy_score(labels, preds)
    }

# ---------------- TRAINING ARGS ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=USE_FP16,
    bf16=USE_BF16,
    logging_dir='./logs',
    logging_steps=50,
    save_total_limit=1,
    dataloader_num_workers=4
)

# ---------------- TRAINER ----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ---------------- TRAIN ----------------
print("Starting training with LoRA...")
trainer.train()

# ---------------- EVAL ----------------
print("Predicting on test set...")
pred_out = trainer.predict(tokenized_test)
logits = pred_out.predictions
if isinstance(logits, tuple):
    logits = logits[0]
preds = np.argmax(logits, axis=-1)
labels = pred_out.label_ids

true_text_labels = [id2label[int(l)] for l in labels]
pred_text_labels = [id2label[int(p)] for p in preds]

print("\nClassification report:")
print(classification_report(labels, preds, target_names=unique_categories, zero_division=0, digits=4))

cm = confusion_matrix(labels, preds, labels=list(range(num_labels)))
cm_df = pd.DataFrame(cm, index=unique_categories, columns=unique_categories)
print("\nConfusion matrix:")
print(cm_df)
