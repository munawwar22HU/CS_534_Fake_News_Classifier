import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

# Load isot dataset
isot_df = pd.read_csv('combined_dataset.csv')  # Adjust the path
isot_df = isot_df[['title', 'text', 'label']]  # Use relevant columns
isot_df['label'] = isot_df['label'].map({1: "Fake", 0: "Real"})  # Convert numeric labels to "Fake" and "Real"

# Verify label consistency
assert set(isot_df['label'].unique()) == {"Fake", "Real"}, "Unexpected Isot labels!"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Prediction function
def predict_fake(title, text):
    input_str = f"<title>{title}<content>{text}<end>"
    input_ids = tokenizer.encode_plus(
        input_str,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
        probabilities = F.softmax(output.logits, dim=1).cpu().numpy()[0]
    return "Fake" if probabilities[0] > probabilities[1] else "Real"

# Evaluation function
def evaluate_model(dataset, dataset_name):
    predictions = []
    for _, row in dataset.iterrows():
        # Handle cases where title or text may be NaN
        title = row['title'] if pd.notna(row['title']) else ""
        text = row['text'] if pd.notna(row['text']) else ""
        predictions.append(predict_fake(title, text))
    
    # Validate predictions and labels
    assert set(predictions).issubset({"Fake", "Real"}), "Unexpected predictions!"
    assert set(dataset['label']).issubset({"Fake", "Real"}), "Unexpected true labels!"
    
    # Generate classification report
    print(f"Evaluation Report for {dataset_name} Dataset")
    print(classification_report(dataset['label'], predictions, target_names=['Fake', 'Real']))

# Evaluate on isot
evaluate_model(isot_df, "isot")
