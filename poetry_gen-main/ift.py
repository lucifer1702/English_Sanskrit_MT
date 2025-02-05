import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.english = dataframe['english'].tolist()
        self.sanskrit = dataframe['sanskrit'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.english)

    def __getitem__(self, idx):
        english_text = str(self.english[idx])
        sanskrit_text = str(self.sanskrit[idx])

        source_encoding = self.tokenizer(
            english_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            sanskrit_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def train_translation_model_with_lora(
    df, 
    model_name="ai4bharat/IndicBART", 
    output_dir="translation_model",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Common attention modules
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameters info

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)


    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)

    def preprocess_function(examples):
        source_lang = examples['english']
        target_lang = examples['sanskrit']
        
        inputs = tokenizer(source_lang, max_length=128, truncation=True, padding='max_length')
        targets = tokenizer(target_lang, max_length=128, truncation=True, padding='max_length')
        
        return {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': targets.input_ids
        }
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        learning_rate=1e-4, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4, 
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True, 
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


if __name__ == "__main__":
    df = pd.read_csv('ift_data.csv')
    model, tokenizer = train_translation_model_with_lora(combined_df)