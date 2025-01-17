import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    df = df.dropna(subset=["Address", "AddressWithCountry", "Country"])
    df["text"] = df["AddressWithCountry"]
    df["Country"] = df["Country"].str.strip().str.lower()

    unique_countries = df["Country"].unique()
    country_to_label = {country: idx for idx, country in enumerate(unique_countries)}

    df["label"] = df["Country"].map(country_to_label)

    return df, country_to_label


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def train_model(dataset, tokenizer, output_dir="models"):
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer))

    train_data, val_data = train_test_split(tokenized_dataset, test_size=0.2, random_state=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(dataset["label"].unique())
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    data_path = "data/house-addresses.csv"
    output_dir = "models"

    df, country_to_label = load_data(data_path)
    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_model(dataset, tokenizer, output_dir)

    print("Training complete. Model saved to", output_dir)
