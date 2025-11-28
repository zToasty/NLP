import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    set_seed, 
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM as NMTModel # Используем T5-подобную модель для аугментации
)
# !!! Импорты PeftModel для продолжения обучения
from peft import LoraConfig, get_peft_model, TaskType, PeftModel 
from sklearn.metrics import f1_score, confusion_matrix
import torch
import os
import random
from tqdm import tqdm

# --- НАСТРОЙКА DEVICE И NMT МОДЕЛЕЙ ДЛЯ АУГМЕНТАЦИИ ---

if torch.cuda.is_available():
    device = torch.device("cuda")
    # Используем BF16, если GPU поддерживает (например, Nvidia Ampere и новее)
    USE_BF16 = torch.cuda.is_bf16_supported() 
else:
    device = torch.device("cpu")
    USE_BF16 = False


# --- Загрузка NMT моделей для Back-Translation (RU -> EN, EN -> RU) ---

NMT_MODEL_RU_EN = None
NMT_MODEL_EN_RU = None

try:
    print("Загрузка NMT-моделей для аугментации...")
    NMT_TOKENIZER_RU_EN = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    NMT_MODEL_RU_EN = NMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)
    
    NMT_TOKENIZER_EN_RU = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    NMT_MODEL_EN_RU = NMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(device)
    print("NMT-модели успешно загружены.")
    
except Exception as e:
    print(f"Внимание: Не удалось загрузить одну или обе NMT-модели. Аугментация Back-Translation будет пропущена. Ошибка: {e}")


# --- ФУНКЦИИ АУГМЕНТАЦИИ ---

def augment_data_for_lora(dataset: Dataset, tokenizer, target_categories: list, num_augmentations_per_sample: int = 2) -> Dataset:
    """
    Аугментирует данные сфокусированным обратным переводом (RU -> EN -> RU)
    для целевых категорий.
    """
    if NMT_MODEL_RU_EN is None or NMT_MODEL_EN_RU is None:
        print("Аугментация пропущена, NMT-модели недоступны.")
        return dataset
        
    print(f"\n--- 1.2. Аугментация {len(target_categories)} проблемных классов ---")
    
    augmented_samples = []
    
    # Фильтруем оригинальные образцы для аугментации
    samples_to_augment = dataset.filter(lambda x: x['category'] in target_categories)
    
    for sample in tqdm(samples_to_augment, desc="Генерация аугментаций"):
        original_text = sample['text']
        original_category = sample['category']
        
        # Добавляем оригинальный сэмпл
        augmented_samples.append(sample)
        
        for _ in range(num_augmentations_per_sample):
            # 1. RU -> EN
            ru_en_tokenized = NMT_TOKENIZER_RU_EN(original_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(device)
            en_tokens = NMT_MODEL_RU_EN.generate(**ru_en_tokenized, max_length=MAX_LENGTH)
            en_text = NMT_TOKENIZER_RU_EN.decode(en_tokens[0], skip_special_tokens=True)
            
            # 2. EN -> RU (Back-Translation)
            en_ru_tokenized = NMT_TOKENIZER_EN_RU(en_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(device)
            ru_tokens = NMT_MODEL_EN_RU.generate(**en_ru_tokenized, max_length=MAX_LENGTH)
            augmented_text = NMT_TOKENIZER_EN_RU.decode(ru_tokens[0], skip_special_tokens=True)

            if augmented_text and augmented_text != original_text:
                augmented_samples.append({
                    'text': augmented_text,
                    'category': original_category
                })

    # Создаем новый Dataset из аугментированных образцов
    augmented_dataset = Dataset.from_list(augmented_samples)
    
    # Объединяем аугментированный набор с остальными не-проблемными образцами
    non_target_samples = dataset.filter(lambda x: x['category'] not in target_categories)
    final_augmented_dataset = concatenate_datasets([non_target_samples, augmented_dataset])
    
    print(f"Оригинальный RU набор: {len(dataset)}")
    print(f"Финальный RU набор после аугментации (всего): {len(final_augmented_dataset)}")
    return final_augmented_dataset


# --- 1. КОНФИГУРАЦИЯ ---
DATASET_NAME = 'Davlan/sib200'
DATASET_LANGUAGE = 'rus_Cyrl'
# Базовая модель (после RU+EN+FRA)
MODEL_PATH = './!!!results_ruT5' 
MAX_LENGTH = 512
MAX_TARGET_LENGTH = 32
# --- ФЛАГИ И ПУТИ ДЛЯ ПРОДОЛЖЕНИЯ ОБУЧЕНИЯ (ОБЯЗАТЕЛЬНО ПРОВЕРИТЬ ПУТЬ) ---
CONTINUE_TRAINING_FROM_LORA = True 
# Путь к папке, куда сохранили адаптеры после первого запуска LoRA (F1 0.9048)
LORA_ADAPTER_PATH = './results_ruT5_lora_aug!!!' 
# --- ПАРАМЕТРЫ ДЛЯ ФИНАЛЬНОГО ДЛИННОГО ЗАПУСКА ---
BATCH_SIZE = 8 
GRADIENT_ACCUMULATION_STEPS = 4 # Эффективный Batch Size = 32
LEARNING_RATE = 3.5e-4  # Снижен для стабильности
NUM_TRAIN_EPOCHS = 100  # Увеличены эпохи для Early Stopping
SEED = 42
# --- КЛАССЫ ДЛЯ АУГМЕНТАЦИИ (travel добавлен) ---
PROBLEM_CLASSES = ['entertainment', 'sports']
AUG_FACTOR = 2 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(f"--- 1. Загрузка данных и подготовка RU-набора для LoRA ---")

train_set_original = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='train')
val_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='validation')
test_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split='test')


# --- 1.2. АУГМЕНТАЦИЯ ---
train_set_augmented = augment_data_for_lora(
    dataset=train_set_original, 
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH),
    target_categories=PROBLEM_CLASSES, 
    num_augmentations_per_sample=AUG_FACTOR
)
combined_train_set = train_set_augmented 

print(f"Финальный размер тренировочного набора для LoRA: {len(combined_train_set)}")

# --- 2. МАППИНГ КАТЕГОРИЙ ---

unique_categories = sorted(list(set(train_set_original['category'])))
print(f"Уникальные категории ({len(unique_categories)}): {unique_categories}")


# --- 3. ТОКЕНИЗАЦИЯ И ПРЕДОБРАБОТКА ---

print("\n--- 3. Токенизация для T5 (Encoder-Decoder) ---")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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

tokenized_train = combined_train_set.map(tokenize_function, batched=True, remove_columns=['text', 'category'])
tokenized_val = val_set.map(tokenize_function, batched=True, remove_columns=['text', 'category'])
tokenized_test = test_set.map(tokenize_function, batched=True, remove_columns=['text', 'category'])

tokenized_train.set_format("torch")
tokenized_val.set_format("torch")
tokenized_test.set_format("torch")


# --- 4. ФУНКЦИИ МЕТРИК С НОРМАЛИЗАЦИЕЙ ---

def normalize_label(label_str):
    """Нормализует метку для сравнения, устраняя различия в пунктуации."""
    if not isinstance(label_str, str):
        return ""
    norm_label = label_str.strip().lower()
    norm_label = norm_label.replace('/', '_').replace('-', '_').replace(' ', '_')
    return norm_label

def compute_metrics_generation(eval_pred):
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    true_labels = [normalize_label(l) for l in decoded_labels]
    predicted_labels = [normalize_label(p) for p in decoded_preds]
    
    valid_categories = [normalize_label(c) for c in unique_categories]
    incorrect_label = 'incorrect_prediction_marker'
    predicted_labels = [p if p in valid_categories else incorrect_label for p in predicted_labels]
    
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    return {
        'macro_f1': macro_f1, 
        'true_labels': true_labels, 
        'predicted_labels': predicted_labels
    }


# --- 5. ИНИЦИАЛИЗАЦИЯ RU-T5 С LoRA (ОБНОВЛЕННЫЙ БЛОК) ---

print("\n--- 5. Инициализация ruT5-large с LoRA ---")

# 1. Загрузка ранее обученной БАЗОВОЙ модели (ruT5-finetuned)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH, 
    load_in_8bit=True,
    device_map="auto"
)

# 2. Конфигурация LoRA (АГРЕССИВНАЯ)
lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo", 'lm_head'], 
    lora_dropout=0.05, 
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM 
)

if CONTINUE_TRAINING_FROM_LORA:
    # 3a. Продолжаем обучение: 
    if os.path.exists(LORA_ADAPTER_PATH):
        print(f"!!! Продолжение обучения. Загрузка адаптеров LoRA из: {LORA_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        
        # 🚨 ИСПРАВЛЕНИЕ: Убедиться, что адаптеры LoRA активированы для обучения
        for name, param in model.named_parameters():
            if 'lora_' in name: # Логика для активации только LoRA-параметров
                param.requires_grad = True
        
    else:
        print(f"!!! ВНИМАНИЕ: Путь {LORA_ADAPTER_PATH} не найден. Начинаем обучение LoRA с нуля.")
        model = get_peft_model(model, lora_config)
else:
    # 3b. Начинаем обучение LoRA с нуля:
    print("!!! Начинаем обучение LoRA с нуля.")
    model = get_peft_model(model, lora_config)

print("LoRA настроена:")
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# --- 6. АРГУМЕНТЫ ТРЕНИРОВКИ ---

print("\n--- 6. Конфигурация Seq2SeqTrainingArguments ---")

training_args = Seq2SeqTrainingArguments( 
    output_dir='./lora_ustal', # Новая папка для финального запуска
    num_train_epochs=NUM_TRAIN_EPOCHS, # 100
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=0.1, 
    learning_rate=LEARNING_RATE, # 3e-4
    weight_decay=0.01, 
    logging_dir='./logs_ruT5_lora_final',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1", 
    greater_is_better=True,
    dataloader_num_workers=4,
    save_total_limit=1,
    fp16=False,
    bf16=USE_BF16, # Используем BF16, если доступно
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
)


# --- 7. ЗАПУСК ТРЕНИРОВКИ С LoRA ---

print("\n--- 7. Запуск тренировки ruT5-large с LoRA ---")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics_generation, 
    tokenizer=tokenizer,
    data_collator=data_collator, 
    # Увеличенное терпение для долгого обучения
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)], 
)

# Тренировка
trainer.train()

# Сохранение только LoRA-адаптера
lora_save_path_final = "./lora_ustal!!!"
print(f"\n--- Сохранение финального LoRA-адаптера в {lora_save_path_final} ---")
model.save_pretrained(lora_save_path_final)
tokenizer.save_pretrained(lora_save_path_final)


# --- 8. ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ С АНАЛИЗОМ ОШИБОК ---

print("\n--- 8. Оценка на тестовом наборе ---")

test_output = trainer.predict(tokenized_test)
test_metrics = test_output.metrics
macro_f1 = test_metrics.get('test_macro_f1', 0)

print(f"Финальные результаты на тестовом наборе:")
print(f"Macro F1: {macro_f1:.4f}")

# 2. Извлекаем истинные и предсказанные строковые метки
true_labels = test_output.metrics['test_true_labels']
predicted_labels = test_output.metrics['test_predicted_labels']

# 3. Анализ Ошибок

print("\n--- АНАЛИЗ ОШИБОК ---")

normalized_unique_categories = sorted(list(set(true_labels)))

if 'incorrect_prediction_marker' in predicted_labels:
    if 'incorrect_prediction_marker' not in normalized_unique_categories:
        normalized_unique_categories.append('incorrect_prediction_marker')


# 3.1. Матрица ошибок (Confusion Matrix)
print("\n### Матрица Ошибок (True vs Predicted) ###")
cm = confusion_matrix(true_labels, predicted_labels, labels=normalized_unique_categories)
cm_df = pd.DataFrame(cm, index=normalized_unique_categories, columns=normalized_unique_categories)

pd.set_option('display.max_columns', None)
print(cm_df)
print("-" * 50)

# 3.2. Вывод примеров с ошибками
print("\n### Топ-10 примеров Ошибок (True != Predicted) ###")

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