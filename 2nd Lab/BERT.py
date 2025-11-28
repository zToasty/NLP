import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    set_seed
)
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torch
import os
import random

DATASET_NAME = 'Davlan/sib200'
DATASET_LANGUAGE = 'rus_Cyrl'
MODEL_PATH = './bert92'

MAX_LENGTH = 512
SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(" 1. Загрузка данных и настройка")

test_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='test')
train_set_original = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='train') 

unique_categories = sorted(list(set(train_set_original['category'])))
label2id = {label: idx for idx, label in enumerate(unique_categories)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"Уникальные категории ({len(unique_categories)}): {unique_categories}")
print(f"Label mapping: {label2id}")


print("\n 2. Загрузка токенизатора и токенизация")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        max_length=MAX_LENGTH, 
        truncation=True, 
        padding=True
    )

def encode_labels(examples):
    examples['labels'] = [label2id[cat] for cat in examples['category']]
    return examples

tokenized_test = test_set.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.map(encode_labels, batched=True)


cols_to_keep = ['input_ids', 'attention_mask', 'labels']
if 'token_type_ids' in tokenized_test.column_names:
    cols_to_keep.append('token_type_ids')

cols_to_remove = [c for c in tokenized_test.column_names if c not in cols_to_keep]
tokenized_test = tokenized_test.remove_columns(cols_to_remove)

tokenized_test.set_format("torch")


print(f"\n 3. Загрузка модели BERT из {MODEL_PATH}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    device_map="auto",
    num_labels=len(unique_categories),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

model.eval()
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics_classification(eval_pred):
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if not isinstance(predictions, np.ndarray):
        predictions = predictions.cpu().numpy()
    if not isinstance(labels, np.ndarray):
        labels = labels.cpu().numpy()

    predicted_classes = np.argmax(predictions, axis=1)

    true_labels_text = [id2label[label] for label in labels]
    predicted_labels_text = [id2label[pred] for pred in predicted_classes]

    def normalize(text):
        if not isinstance(text, str): return ""
        text = text.strip().lower()
        return text.replace('/', '_').replace('-', '_').replace(' ', '_')

    true_labels_norm = [normalize(l) for l in true_labels_text]
    predicted_labels_norm = [normalize(p) for p in predicted_labels_text]

    macro_f1 = f1_score(true_labels_norm, predicted_labels_norm, average='macro', zero_division=0)

    target_names_norm = [normalize(c) for c in unique_categories]

    report_str = classification_report(
        true_labels_norm, predicted_labels_norm, 
        labels=target_names_norm,
        zero_division=0,
        digits=4
    )

    return {
        'macro_f1': macro_f1,
        'true_labels': true_labels_norm,
        'predicted_labels': predicted_labels_norm,
        'true_labels_raw': true_labels_text,
        'predicted_labels_raw': predicted_labels_text,
        'classification_report_str': report_str,
        'predictions_logits': predictions
    }


print("\n 4. Запуск оценки")

eval_args = TrainingArguments(
    output_dir='./temp_eval_bert',
    per_device_eval_batch_size=16,
    dataloader_num_workers=4,
    bf16=True if torch.cuda.is_available() else False,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=eval_args,
    compute_metrics=compute_metrics_classification,
    data_collator=data_collator,
)

test_output = trainer.predict(tokenized_test)

metrics_res = compute_metrics_classification((test_output.predictions, test_output.label_ids))

final_f1 = metrics_res['macro_f1']
true_labels = metrics_res['true_labels']
pred_labels = metrics_res['predicted_labels']
true_labels_raw = metrics_res['true_labels_raw']
pred_labels_raw = metrics_res['predicted_labels_raw']
classification_report_str = metrics_res['classification_report_str']

print(f"\n RESULT MACRO F1 (BERT): {final_f1:.4f}")

print("\n 5. CLASSIFICATION REPORT")
print(classification_report_str)


print("\n 6. Матрица ошибок")

labels_list = sorted(list(set(true_labels)))
cm = confusion_matrix(true_labels, pred_labels, labels=labels_list)
cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(cm_df)


print("\n 7. Топ-20 примеров Ошибок")
error_count = 0
for i in range(len(true_labels)):
    if true_labels[i] != pred_labels[i]:
        if error_count >= 20:
            break
        original_text = test_set[i]['text']

        print(f"True: {true_labels_raw[i]} | Pred: {pred_labels_raw[i]}")
        print(f"Text: {original_text[:150]}...")
        print("-" * 50)
        error_count += 1

print(f"\nВсего ошибок в тестовой выборке: {sum(1 for t, p in zip(true_labels, pred_labels) if t != p)}")
print(f"Общий размер тестовой выборки: {len(true_labels)}")
