import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import torch

# Load WELFake dataset
welfake_df = pd.read_csv('combined_dataset.csv')  # Adjust the path
welfake_df = welfake_df[['title', 'text', 'label']]  # Use relevant columns
welfake_df['label'] = welfake_df['label'].map({1: 1, 0: 0})  # Ensure labels are numeric

# Drop rows with missing title or text
welfake_df.dropna(subset=['text'], inplace=True)
welfake_df.fillna({'title': ''}, inplace=True)  # Replace missing titles with empty strings

# Combine title and text
welfake_df['content'] = welfake_df['title'] + " " + welfake_df['text']

# Define compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(welfake_df)):
    print(f"\nTraining fold {fold + 1}/5")

    train_texts = welfake_df.iloc[train_idx]['content'].tolist()
    val_texts = welfake_df.iloc[val_idx]['content'].tolist()
    train_labels = welfake_df.iloc[train_idx]['label'].tolist()
    val_labels = welfake_df.iloc[val_idx]['label'].tolist()

    # Load the BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Tokenize data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    # Convert data to PyTorch datasets
    class FakeNewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = FakeNewsDataset(train_encodings, train_labels)
    val_dataset = FakeNewsDataset(val_encodings, val_labels)

    # Training arguments with gradient clipping and learning rate scheduling
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold + 1}",          # Output directory
        evaluation_strategy="epoch",    # Evaluate every epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir=f"./logs_fold_{fold + 1}",           # Directory for logs
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        gradient_accumulation_steps=2,    # Gradient accumulation for larger effective batch size
        max_grad_norm=1.0,                # Gradient clipping
        lr_scheduler_type="linear",     # Linear learning rate scheduler
        warmup_ratio=0.1,                 # Warm-up steps as a ratio of total training steps
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"\nEvaluation Results for Fold {fold + 1}:")
    print(eval_results)
    fold_results.append(eval_results)

# Average results across folds
avg_accuracy = sum([result['eval_accuracy'] for result in fold_results]) / len(fold_results)
avg_precision = sum([result['eval_precision'] for result in fold_results]) / len(fold_results)
avg_recall = sum([result['eval_recall'] for result in fold_results]) / len(fold_results)
avg_f1 = sum([result['eval_f1'] for result in fold_results]) / len(fold_results)

print("\nCross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1 Score: {avg_f1}")