import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, # Изменено для T5
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, # Изменено для T5
    set_seed, 
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
import torch
from collections import Counter
import os
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


# --- 1. КОНФИГУРАЦИЯ ---
DATASET_NAME = 'Davlan/sib200'
DATASET_LANGUAGE = 'rus_Cyrl'
MODEL_CHECKPOINT = 'ai-forever/ruT5-large'
MAX_LENGTH = 512 # Максимальная длина входа (Encoder)
MAX_TARGET_LENGTH = 32 # Максимальная длина выхода 
BATCH_SIZE = 4 
GRADIENT_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 4e-5
NUM_TRAIN_EPOCHS = 10 
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(" 1. Загрузка данных и мультиязыковое объединение ")

# Загрузка и объединение данных
train_set_original = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='train')
val_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='validation')
test_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='test')

eng_train = load_dataset(DATASET_NAME, 'eng_Latn', split='train')
fra_train = load_dataset(DATASET_NAME, 'fra_Latn', split='train')

# Объединяем тренировочные данные
combined_train_set = concatenate_datasets([train_set_original, eng_train, fra_train])

unique_categories = sorted(list(set(train_set_original['category'])))
print(f"Уникальные категории ({len(unique_categories)}): {unique_categories}")
print(len(combined_train_set))


print("\n 3. Токенизация для T5 (Encoder-Decoder)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples['text'], 
        max_length=MAX_LENGTH, 
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['category'],
            max_length=MAX_TARGET_LENGTH, 
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Применение токенизации
tokenized_train = combined_train_set.map(tokenize_function, batched=True, remove_columns=['text', 'category'])
tokenized_val = val_set.map(tokenize_function, batched=True, remove_columns=['text', 'category'])
tokenized_test = test_set.map(tokenize_function, batched=True, remove_columns=['text', 'category'])

tokenized_train.set_format("torch")
tokenized_val.set_format("torch")
tokenized_test.set_format("torch")

print("\n 4. Инициализация ruT5-large")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def normalize_label(label_str):
    """Нормализует метку для сравнения, устраняя различия в пунктуации."""
    if not isinstance(label_str, str):
        return ""
    # 1. Приведение к нижнему регистру, удаление пробелов
    norm_label = label_str.strip().lower()
    # 2. Замена неоднозначных символов на подчеркивание
    norm_label = norm_label.replace('/', '_').replace('-', '_').replace(' ', '_')
    return norm_label

def compute_metrics_generation(eval_pred):
    from sklearn.metrics import f1_score
    
    predictions, labels = eval_pred
    
    # 1. Декодируем сгенерированные ID обратно в текст
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 2. Заменяем -100 и декодируем истинные метки
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 3. Нормализация строк и расчет F1
    
    true_labels = [normalize_label(l) for l in decoded_labels]
    predicted_labels = [normalize_label(p) for p in decoded_preds]
    
    # Нормализация списка всех возможных категорий для фильтрации
    valid_categories = [normalize_label(c) for c in unique_categories]
    
    # Если прогноз не совпадает ни с одной нормализованной категорией, считаем его неверным
    incorrect_label = 'incorrect_prediction_marker'
    predicted_labels = [p if p in valid_categories else incorrect_label for p in predicted_labels]
    
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    return {
        'macro_f1': macro_f1,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }



print("\n 5. Конфиг")

training_args = Seq2SeqTrainingArguments( 
    output_dir='./results_ruT5',
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=0.06, 
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir='./logs_ruT5',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1", 
    greater_is_better=True,
    dataloader_num_workers=4,
    save_total_limit=1,
    fp16=False,
    bf16=True,
    
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
)


print("\n 6. Запуск тренировки ruT5-large")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics_generation,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

trainer.train()


print("\n 7. Оценка на тестовом наборе")

test_output = trainer.predict(tokenized_test)
test_metrics = test_output.metrics
macro_f1 = test_metrics.get('test_macro_f1', 0)

print("Финальные результаты на тестовом наборе:")
print(f"Macro F1: {macro_f1:.4f}")


true_labels = test_output.label_ids['true_labels']
predicted_labels = test_output.label_ids['predicted_labels']


print("\n--- АНАЛИЗ ОШИБОК ---")

# Получаем уникальные нормализованные метки для осей матрицы
normalized_unique_categories = sorted(list(set(true_labels)))

# Если метка 'incorrect_prediction_marker' присутствует, добавляем ее в метки
if 'incorrect_prediction_marker' in predicted_labels:
    if 'incorrect_prediction_marker' not in normalized_unique_categories:
        normalized_unique_categories.append('incorrect_prediction_marker')


# 3.1. Матрица ошибок (Confusion Matrix)
print("\n### Матрица Ошибок (True vs Predicted) ###")
cm = confusion_matrix(true_labels, predicted_labels, labels=normalized_unique_categories)
cm_df = pd.DataFrame(cm, index=normalized_unique_categories, columns=normalized_unique_categories)

print(cm_df)
print("-" * 50)

original_test_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='test')
error_count = 0

for i in range(len(true_labels)):
    if true_labels[i] != predicted_labels[i]:
        if error_count >= 10:
            break
        
        original_text = original_test_set[i]['text']
        
        print(f"--- Ошибка №{error_count + 1} ---")
        print(f"ИСТИНА: {true_labels[i]}")
        print(f"ПРОГНОЗ: {predicted_labels[i]}")
        print(f"ТЕКСТ: {original_text[:150]}...")
        error_count += 1
        
print("-" * 50)