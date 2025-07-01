import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Layer
import tensorflow.keras.backend as K
from datasets import load_dataset
import numpy as np

# --- Sidebar Interactive Controls ---
st.sidebar.title("🛠️ Tuning Panel")
SEQ_LEN = st.sidebar.slider("Sequence Length", 5, 30, 10, step=1)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
fraud_weight = st.sidebar.slider("Fraud Class Weight", 1.0, 20.0, 5.0, step=0.5)
pred_threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.5, step=0.01)
use_focal_loss = st.sidebar.checkbox("Use Focal Loss", value=False)
layer_choice = st.sidebar.selectbox("Model Depth", [1, 2, 3], index=1)
use_hour_feature = st.sidebar.checkbox("Include Hour Feature", value=True)

# --- Load Data ---
st.title("💳 Credit Card Fraud Detection with Bi-LSTM + Attention")
st.markdown("---")

@st.cache_data
def load_data():
    dataset = load_dataset("dazzle-nu/CIS435-CreditCardFraudDetection")
    return dataset["train"].to_pandas()

data = load_data().sample(n=50000, random_state=42).reset_index(drop=True)
data = data.sort_values(['cc_num', 'unix_time'])
if use_hour_feature:
    data['hour'] = pd.to_datetime(data['unix_time'], unit='s').dt.hour

features = ['amt', 'city_pop', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long']
if use_hour_feature:
    features.append('hour')

X = data[features].values
y = data['is_fraud'].values
cc_nums = data['cc_num'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_seqs, y_seqs = [], []
for cc in np.unique(cc_nums):
    idxs = np.where(cc_nums == cc)[0]
    card_X = X_scaled[idxs]
    card_y = y[idxs]
    if len(card_X) >= SEQ_LEN:
        for i in range(len(card_X) - SEQ_LEN + 1):
            X_seqs.append(card_X[i:i+SEQ_LEN])
            y_seqs.append(card_y[i+SEQ_LEN-1])

X_seqs = np.array(X_seqs)
y_seqs = np.array(y_seqs)

st.write(f"Total sequences: {len(X_seqs)} (Sequence length: {SEQ_LEN})")

X_train, X_test, y_train, y_test = train_test_split(X_seqs, y_seqs, test_size=0.2, stratify=y_seqs, random_state=42)

# --- Attention Layer ---
class Attention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.return_attention = return_attention

    def call(self, inputs):
        score = K.softmax(K.sum(inputs, axis=-1), axis=-1)
        score = K.expand_dims(score)
        attended = K.sum(inputs * score, axis=1)
        return (attended, score) if self.return_attention else attended

# --- Build Model ---
input_shape = (SEQ_LEN, len(features))
inputs = Input(shape=input_shape)
x = inputs
for i in range(layer_choice):
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
x = Attention()(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs, output)

# --- Optional Focal Loss ---
if use_focal_loss:
    from keras_losses import BinaryFocalLoss
    loss_fn = BinaryFocalLoss(gamma=2)
else:
    loss_fn = 'binary_crossentropy'

model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

# --- Train Model ---
from sklearn.utils import class_weight
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: 1.0, 1: fraud_weight}

with st.spinner("Training model..."):
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        class_weight=class_weights,
        verbose=0
    )

# --- Predict ---
with st.spinner("Predicting..."):
    y_proba = model.predict(X_test).flatten()
y_pred = (y_proba > pred_threshold).astype(int)

# --- Evaluation ---
st.subheader("📉 Training History")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='Train Acc')
ax[0].plot(history.history['val_accuracy'], label='Val Acc')
ax[0].set_title('Accuracy')
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Val Loss')
ax[1].set_title('Loss')
ax[1].legend()
st.pyplot(fig)

st.subheader("📈 Classification Report")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().round(2))

st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm, cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
    ax_cm.text(j, i, str(z), ha='center', va='center')
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel("FPR")
ax_roc.set_ylabel("TPR")
ax_roc.legend()
st.pyplot(fig_roc)

st.subheader("🔍 Precision, Recall, F1-score (Bar Chart)")
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
labels = ['Not Fraud', 'Fraud']
x = np.arange(len(labels))
fig_bar, ax_bar = plt.subplots()
width = 0.25
ax_bar.bar(x - width, prec, width, label='Precision')
ax_bar.bar(x, rec, width, label='Recall')
ax_bar.bar(x + width, f1, width, label='F1-score')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels)
ax_bar.set_ylim(0, 1)
ax_bar.legend()
st.pyplot(fig_bar)

st.subheader("📈 Fraud Probability Histogram")
fig_prob, ax_prob = plt.subplots()
ax_prob.hist(y_proba[y_test==0], bins=50, alpha=0.6, label="Non-Fraud", density=True)
ax_prob.hist(y_proba[y_test==1], bins=50, alpha=0.6, label="Fraud", density=True)
ax_prob.set_xlabel("Predicted Fraud Probability")
ax_prob.legend()
st.pyplot(fig_prob)

# Attention Heatmap
st.subheader("🧠 Attention Heatmap (Sample Fraud Sequence)")
attention_model = Model(inputs=model.input, outputs=model.layers[-2].output)
fraud_idxs = np.where(y_test == 1)[0]
if len(fraud_idxs) > 0:
    fraud_sample = X_test[fraud_idxs[0:1]]
    attention_layer = model.get_layer(index=-2)
    att_output = attention_layer(fraud_sample).numpy()
    fig_att, ax_att = plt.subplots()
    im = ax_att.imshow([att_output[0]], cmap='Reds', aspect='auto')
    ax_att.set_title("Attention Weights")
    st.pyplot(fig_att)

# Feature Importance Proxy
st.subheader("📊 Feature Importance Proxy (Mean Absolute Value)")
feat_means = np.mean(np.abs(X_test), axis=(0,1))
fig_feat, ax_feat = plt.subplots()
ax_feat.bar(features, feat_means)
ax_feat.set_ylabel("Mean Abs. Value")
ax_feat.set_title("Feature Importance Proxy")
st.pyplot(fig_feat)

# Save CSV
st.subheader("💾 Download 100 Sample Predictions")
sample_idx = np.random.choice(np.arange(X_test.shape[0]), 100, replace=False)
sample_data = X_test[sample_idx].reshape(100, SEQ_LEN * len(features))
sample_pred = y_pred[sample_idx]
sample_prob = y_proba[sample_idx]
sample_df = pd.DataFrame(sample_data, columns=[f"{f}_t{t+1}" for t in range(SEQ_LEN) for f in features])
sample_df['Predicted Fraud'] = sample_pred
sample_df['Fraud Probability'] = sample_prob
csv = sample_df.to_csv(index=False).encode()
st.download_button("Download CSV", data=csv, file_name="fraud_predictions.csv")

st.caption("Model: Bi-LSTM + Attention | Interactive Fraud Detection | Author: Amirol Ahmad")
