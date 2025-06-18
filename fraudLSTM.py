import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Layer
import tensorflow.keras.backend as K
from datasets import load_dataset
import numpy as np

# Step 1: Load real data from Hugging Face dataset
@st.cache_data
def load_data():
    dataset = load_dataset("dazzle-nu/CIS435-CreditCardFraudDetection")
    df = dataset["train"].to_pandas()
    return df

data = load_data()
data = data.sample(n=100000, random_state=42).reset_index(drop=True)

# Show column names for debugging
# st.write("### 📋 Columns in Dataset")
# st.write(data.columns.tolist())

# Step 2: Prepare data for Bi-LSTM + Attention
features = ['amt', 'city_pop', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long']
X = data[features]
y = data['is_fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Define Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def call(self, inputs):
        score = K.softmax(K.sum(inputs, axis=-1), axis=-1)
        score = K.expand_dims(score)
        return K.sum(inputs * score, axis=1)

# Step 4: Build Bi-LSTM Model with Attention
input_shape = (X_train.shape[1], X_train.shape[2])
inputs = Input(shape=input_shape)
x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Attention()(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs, output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    class_weight=class_weights
)

# Step 5: Streamlit App
st.title("💳 Credit Card Fraud Detection with Bi-LSTM + Attention")
st.write("Using a stacked Bidirectional LSTM with Attention to detect fraudulent transactions.")

st.subheader("📄 Sample Predictions")
# Force sample: 5 fraud + 5 non-fraud from test set
y_test_array = y_test.to_numpy()
fraud_idx = (y_test_array == 1).nonzero()[0][:5]
nonfraud_idx = (y_test_array == 0).nonzero()[0][:5]
combined_idx = list(fraud_idx) + list(nonfraud_idx)

sample_data = X_test[combined_idx]
sample_df = pd.DataFrame(sample_data.reshape(10, X.shape[1]), columns=X.columns)
predictions = (model.predict(sample_data) > 0.5).astype(int).flatten()
sample_df['Predicted Fraud'] = predictions

st.dataframe(sample_df)

# Summary stats
fraud_count = sample_df['Predicted Fraud'].value_counts()
st.subheader("🔢 Fraud Prediction Summary")
st.write(fraud_count)

# Pie chart
st.subheader("📊 Fraud Pie Chart")
fig, ax = plt.subplots()
fraud_count.plot.pie(autopct='%1.1f%%', ax=ax, title='Fraud vs Non-Fraud')
st.pyplot(fig)

# Step 6: Show Training History
st.subheader("📉 Training Performance")
fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 4))
ax_hist[0].plot(history.history['accuracy'], label='Train Accuracy')
ax_hist[0].plot(history.history['val_accuracy'], label='Val Accuracy')
ax_hist[0].set_title('Accuracy per Epoch')
ax_hist[0].legend()

ax_hist[1].plot(history.history['loss'], label='Train Loss')
ax_hist[1].plot(history.history['val_loss'], label='Val Loss')
ax_hist[1].set_title('Loss per Epoch')
ax_hist[1].legend()

st.pyplot(fig_hist)

# Step 7: Model Evaluation
st.subheader("📈 Model Evaluation")
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
st.text(classification_report(y_test, y_pred))

# Step 8: Save predictions to CSV
st.subheader("💾 Save Predictions")
if st.button("Download CSV"):
    sample_df.to_csv("lstm_fraud_predictions.csv", index=False)
    st.success("Predictions saved to lstm_fraud_predictions.csv")

st.markdown("---")
st.text("Model: Bi-LSTM + Attention | Data: dazzle-nu/CIS435-CreditCardFraudDetection | Author: Amirol Ahmad | License: MIT")
