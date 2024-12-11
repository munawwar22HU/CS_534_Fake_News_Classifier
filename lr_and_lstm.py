# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# # Set GPU configuration
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
#         tf.config.set_logical_device_configuration(
#             device, [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
#         )
# print("Using TensorFlow version:", tf.__version__)
# print("GPU Devices:", physical_devices)

# # Load dataset
# welfake = pd.read_csv("combined_dataset.csv")

# # Preprocessing function for LSTM
# def preprocess_data_lstm(df, max_words=20000, max_len=300):
#     df = df.dropna(subset=['text', 'label'])
#     df['text'] = df['text'].astype(str)
    
#     # Encode labels
#     label_encoder = LabelEncoder()
#     df['label'] = label_encoder.fit_transform(df['label'])
    
#     # Tokenize text
#     tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
#     tokenizer.fit_on_texts(df['text'])
#     sequences = tokenizer.texts_to_sequences(df['text'])
#     padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
#     return padded_sequences, df['label'], tokenizer

# # Preprocessing function for Logistic Regression
# def preprocess_data_logistic(df):
#     df = df.dropna(subset=['text', 'label'])
#     df['text'] = df['text'].astype(str)
    
#     # Encode labels
#     label_encoder = LabelEncoder()
#     df['label'] = label_encoder.fit_transform(df['label'])
    
#     return df['text'], df['label']

# # Preprocess WELFake data
# x_welfake_lstm, y_welfake, tokenizer = preprocess_data_lstm(welfake)
# x_welfake_lr, y_welfake_lr = preprocess_data_logistic(welfake)

# # Split WELFake data into train and test
# x_train_lstm, x_test_welfake_lstm, y_train, y_test_welfake = train_test_split(x_welfake_lstm, y_welfake, test_size=0.2, random_state=42)
# x_train_lr, x_test_welfake_lr, y_train_lr, y_test_welfake_lr = train_test_split(x_welfake_lr, y_welfake_lr, test_size=0.2, random_state=42)

# # Logistic Regression Model
# print("\nTraining Logistic Regression Model")
# vectorizer = TfidfVectorizer(max_features=20000)
# x_train_lr_tfidf = vectorizer.fit_transform(x_train_lr)
# x_test_welfake_lr_tfidf = vectorizer.transform(x_test_welfake_lr)

# logistic_model = LogisticRegression(max_iter=1000)
# logistic_model.fit(x_train_lr_tfidf, y_train_lr)

# # Evaluate Logistic Regression on WELFake test data
# print("\nEvaluation on WELFake Test Data (Logistic Regression)")
# welfake_test_predictions_lr = logistic_model.predict(x_test_welfake_lr_tfidf)
# print(classification_report(y_test_welfake_lr, welfake_test_predictions_lr))

# # Define LSTM model
# def build_lstm_model(vocab_size, embedding_dim=128, max_len=300):
#     model = Sequential([
#         Embedding(vocab_size, embedding_dim, input_length=max_len),
#         LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Build LSTM model
# vocab_size = len(tokenizer.word_index) + 1
# lstm_model = build_lstm_model(vocab_size)

# # Early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Train LSTM model
# print("\nTraining LSTM Model")
# lstm_model.fit(x_train_lstm, y_train, validation_split=0.2, epochs=5, batch_size=64, 
#                callbacks=[early_stopping])

# # Evaluate LSTM on WELFake test data
# print("\nEvaluation on WELFake Test Data (LSTM)")
# welfake_test_predictions_lstm = lstm_model.predict(x_test_welfake_lstm)
# welfake_test_predictions_lstm = (welfake_test_predictions_lstm > 0.5).astype(int)
# print(classification_report(y_test_welfake, welfake_test_predictions_lstm))


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_logical_device_configuration(
            device, [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
        )
print("Using TensorFlow version:", tf.__version__)
print("GPU Devices:", physical_devices)

# Load dataset
welfake = pd.read_csv("combined_dataset.csv")

# Preprocessing function for LSTM
def preprocess_data_lstm(df, max_words=20000, max_len=300):
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded_sequences, df['label'], tokenizer

# Preprocessing function for Logistic Regression
def preprocess_data_logistic(df):
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    return df['text'], df['label']

# Preprocess WELFake data
x_welfake_lstm, y_welfake, tokenizer = preprocess_data_lstm(welfake)
x_welfake_lr, y_welfake_lr = preprocess_data_logistic(welfake)

# Split WELFake data into train and test
x_train_lstm, x_test_welfake_lstm, y_train, y_test_welfake = train_test_split(x_welfake_lstm, y_welfake, test_size=0.2, random_state=42)
x_train_lr, x_test_welfake_lr, y_train_lr, y_test_welfake_lr = train_test_split(x_welfake_lr, y_welfake_lr, test_size=0.2, random_state=42)

# Logistic Regression Model
print("\nTraining Logistic Regression Model")
vectorizer = TfidfVectorizer(max_features=20000)
x_train_lr_tfidf = vectorizer.fit_transform(x_train_lr)
x_test_welfake_lr_tfidf = vectorizer.transform(x_test_welfake_lr)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(x_train_lr_tfidf, y_train_lr)

# Training accuracy for Logistic Regression
lr_train_predictions = logistic_model.predict(x_train_lr_tfidf)
lr_train_accuracy = accuracy_score(y_train_lr, lr_train_predictions)
print(f"Logistic Regression Training Accuracy: {lr_train_accuracy:.4f}")

# Testing accuracy for Logistic Regression
lr_test_predictions = logistic_model.predict(x_test_welfake_lr_tfidf)
lr_test_accuracy = accuracy_score(y_test_welfake_lr, lr_test_predictions)
print(f"Logistic Regression Testing Accuracy: {lr_test_accuracy:.4f}")

# Print classification report
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test_welfake_lr, lr_test_predictions))

# Define LSTM model
def build_lstm_model(vocab_size, embedding_dim=128, max_len=300):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build LSTM model
vocab_size = len(tokenizer.word_index) + 1
lstm_model = build_lstm_model(vocab_size)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train LSTM model
print("\nTraining LSTM Model")
history = lstm_model.fit(x_train_lstm, y_train, validation_split=0.2, epochs=5, batch_size=64, 
                         callbacks=[early_stopping])

# Training accuracy for LSTM
train_loss, train_accuracy = lstm_model.evaluate(x_train_lstm, y_train, verbose=0)
print(f"LSTM Training Accuracy: {train_accuracy:.4f}")

# Testing accuracy for LSTM
test_loss, test_accuracy = lstm_model.evaluate(x_test_welfake_lstm, y_test_welfake, verbose=0)
print(f"LSTM Testing Accuracy: {test_accuracy:.4f}")

# Print classification report
welfake_test_predictions_lstm = lstm_model.predict(x_test_welfake_lstm)
welfake_test_predictions_lstm = (welfake_test_predictions_lstm > 0.5).astype(int)
print("\nClassification Report (LSTM):")
print(classification_report(y_test_welfake, welfake_test_predictions_lstm))
