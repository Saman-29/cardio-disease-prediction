
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight

# Load structured data (Heart Disease dataset from UCI Repository)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]
heart_data = pd.read_csv(url, names=column_names, na_values='?', header=None)
heart_data.dropna(inplace=True)

# Define structured features and target
structured_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
y = (heart_data['num'] > 0).astype(int)  # Binary classification: 0 for no disease, 1 for disease
X_structured = heart_data[structured_features]

# Simulate unstructured text data (clinical notes)
text_data = [
    "Patient experiences chest pain with high cholesterol and elevated blood pressure.",
    "Healthy individual with normal blood pressure and no signs of cardiovascular disease.",
    "Patient has a history of smoking, diabetes, and occasional chest discomfort.",
    "Active lifestyle, no significant health issues, normal cholesterol levels."
] * (len(X_structured) // 4)

# Ensure text data length matches structured data
if len(text_data) > len(X_structured):
    text_data = text_data[:len(X_structured)]
elif len(text_data) < len(X_structured):
    text_data = text_data * (len(X_structured) // len(text_data)) + text_data[:(len(X_structured) % len(text_data))]

# Tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text data
max_len = 50
X_text = tokenizer(
    text_data,
    padding='max_length',
    truncation=True,
    max_length=max_len,
    return_tensors='tf'
)

# Convert tensors to NumPy arrays
X_text_input_ids = X_text['input_ids'].numpy()

# Split data into training and test sets
X_train_structured, X_test_structured, X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_structured, X_text_input_ids, y, test_size=0.2, random_state=42
)

# Standardize structured data
scaler = StandardScaler()
X_train_structured = scaler.fit_transform(X_train_structured)
X_test_structured = scaler.transform(X_test_structured)

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model definition using TFBertForSequenceClassification
structured_input = tf.keras.layers.Input(shape=(len(structured_features),), name='structured_input')
text_input = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='text_input')

# Use BERT for classification (pre-trained)
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Freeze all layers except the last layers
for layer in bert_model.layers[:-1]:
    layer.trainable = False

bert_output = bert_model(text_input).logits

# Dense layers for structured input with Batch Normalization
x_structured = tf.keras.layers.Dense(16, activation='relu')(structured_input)
x_structured = tf.keras.layers.BatchNormalization()(x_structured)
x_structured = tf.keras.layers.Dense(8, activation='relu')(x_structured)

# Combine structured and text features
combined = tf.keras.layers.concatenate([x_structured, bert_output])
combined = tf.keras.layers.Dropout(0.3)(combined)  # Dropout to prevent overfitting
output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

# Build and compile the model
model = tf.keras.Model(inputs=[structured_input, text_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with class weights
history = model.fit(
    [X_train_structured, X_train_text], y_train,
    validation_data=([X_test_structured, X_test_text], y_test),
    epochs=20,  # Increased epochs to allow early stopping to take effect
    batch_size=16,
    callbacks=[early_stopping],
    class_weight=class_weight_dict  # Apply class weights to handle imbalance
)

# Evaluate the model
predictions = model.predict([X_test_structured, X_test_text]) > 0.5
print("Classification Report:")
print(classification_report(y_test, predictions))
print(f"ROC-AUC Score: {roc_auc_score(y_test, predictions)}")
