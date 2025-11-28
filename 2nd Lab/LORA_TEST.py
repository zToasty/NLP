import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import PeftModel 
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torch
import os
import random

DATASET_NAME = 'Davlan/sib200'
DATASET_LANGUAGE = 'rus_Cyrl'


MODEL_PATH = './!!!results_ruT5'
LORA_PATH = './results_ruT5_lora_aug!!!' 

MAX_LENGTH = 512
MAX_TARGET_LENGTH = 32
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
print(f"Уникальные категории ({len(unique_categories)}): {unique_categories}")

print("\n 2. Загрузка токенизатора и токенизация")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize_function(examples):
    model_inputs = tokenizer(examples['text'], max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(text_target=examples['category'], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_test = test_set.map(tokenize_function, batched=True, remove_columns=['text', 'category'])
tokenized_test.set_format("torch")


print(f"\n 3. Загрузка Base  из {MODEL_PATH} и наложение LoRA из {LORA_PATH}")

base_ft_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(base_ft_model, LORA_PATH)
model.eval()
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def compute_metrics_generation(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple): predictions = predictions[0]
    
    if not isinstance(predictions, np.ndarray): predictions = predictions.cpu().numpy()
    if not isinstance(labels, np.ndarray): labels = labels.cpu().numpy()

    vocab_limit = tokenizer.vocab_size 
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    predictions = np.where(predictions >= vocab_limit, pad_id, predictions)
    labels = np.where(labels != -100, labels, pad_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def normalize(text):
        if not isinstance(text, str): return ""
        text = text.strip().lower()
        return text.replace('/', '_').replace('-', '_').replace(' ', '_')

    true_labels = [normalize(l) for l in decoded_labels]
    raw_normalized_preds = [normalize(p) for p in decoded_preds]
    
    valid = set([normalize(c) for c in unique_categories])
    final_preds = [p if p in valid else 'incorrect' for p in raw_normalized_preds]

    macro_f1 = f1_score(true_labels, final_preds, average='macro', zero_division=0)
    
    target_names_norm = [normalize(c) for c in unique_categories]
    
    report = classification_report(
        true_labels, 
        final_preds, 
        labels=target_names_norm,
        output_dict=True, 
        zero_division=0
    )
    
    report_str = classification_report(
        true_labels, 
        final_preds, 
        labels=target_names_norm, 
        zero_division=0,
        digits=4
    )

    return {
        'macro_f1': macro_f1,
        'true_labels': true_labels,
        'predicted_labels': final_preds,
        'raw_predictions': raw_normalized_preds,
        'classification_report_str': report_str
    }


print("\n 4. Запуск оценки")

eval_args = Seq2SeqTrainingArguments( 
    output_dir='./temp_eval_combined',
    per_device_eval_batch_size=16, 
    dataloader_num_workers=4,
    bf16=True, 
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=eval_args,
    compute_metrics=compute_metrics_generation,
    data_collator=data_collator, 
)

test_output = trainer.predict(tokenized_test)
metrics_res = compute_metrics_generation((test_output.predictions, test_output.label_ids))

final_f1 = metrics_res['macro_f1']
true_labels = metrics_res['true_labels']
pred_labels = metrics_res['predicted_labels']
raw_preds = metrics_res['raw_predictions']
classification_report_str = metrics_res['classification_report_str']

print(f"\n LORA RESULT MACRO F1: {final_f1:.4f}")

print("\n 5. АНАЛИЗ МЕТРИК ПО КЛАССАМ")
print(classification_report_str)

print("\n 6. АНАЛИЗ МАТРИЦЫ И ОШИБОК")

print("\n Матрица Ошибок")
labels_list = sorted(list(set(true_labels)))
if 'incorrect' in pred_labels and 'incorrect' not in labels_list:
    labels_list.append('incorrect')

cm = confusion_matrix(true_labels, pred_labels, labels=labels_list)
cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(cm_df)

print("\n Топ-20 примеров Ошибок")

error_count = 0
print("Формат: Истина | Что модель реально сказала | Текст")

for i in range(len(true_labels)):
    if true_labels[i] != pred_labels[i]:
        if error_count >= 20:
            break
        
        original_text = test_set[i]['text']
        model_output = raw_preds[i] if raw_preds[i] else "<EMPTY>"
        
        print(f"True: {true_labels[i]} | Raw Pred: {model_output}")
        print(f"Text: {original_text[:100]}...")
        print("-" * 20)
        error_count += 1