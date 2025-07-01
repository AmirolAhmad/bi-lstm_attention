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
st.sidebar.title("🛠️ Tuning Model")
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

with st.spinner("Loading data... Please wait"):
    data = load_data()
st.success("Data loaded!")
st.caption("Dataset contains credit card transactions with fraud labels from HuggingFace - dazzle-nu/CIS435-CreditCardFraudDetection.")

data = data.sample(n=50000, random_state=42).reset_index(drop=True)
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

# Simpan index asal sebelum split
original_indices = np.arange(len(y_seqs))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_seqs, y_seqs, original_indices, test_size=0.2, stratify=y_seqs, random_state=42
)

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
st.success("Training complete!")

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

# Card Transaction Timeline Explorer
st.subheader("🔎 Card Transaction Timeline Explorer")
card_ids = data['cc_num'].unique()
selected_card = st.selectbox("Select cc_num (card)", card_ids)
if selected_card:
    with st.spinner(f"Loading transactions for {selected_card}..."):
        card_txn = data[data['cc_num'] == selected_card].sort_values('unix_time')
        fig_timeline, ax_timeline = plt.subplots(figsize=(10, 2))
        ax_timeline.plot(card_txn['unix_time'], card_txn['amt'], marker='o', label='Amount', color='blue')
        fraud_txn = card_txn[card_txn['is_fraud'] == 1]
        ax_timeline.scatter(fraud_txn['unix_time'], fraud_txn['amt'], color='red', label='Fraud', s=80, zorder=2)
        ax_timeline.set_xlabel("Unix Time")
        ax_timeline.set_ylabel("Amount")
        ax_timeline.set_title(f"Transaction Timeline for Card: {selected_card}")
        ax_timeline.legend()
        st.pyplot(fig_timeline)
        st.dataframe(card_txn[['unix_time', 'amt', 'is_fraud', 'merchant', 'city', 'state', 'zip']])

# Per-Card Fraud Risk Overview
st.subheader("📊 Per-Card Fraud Risk Overview")
risk_df = data.groupby('cc_num')['is_fraud'].agg(['count', 'sum'])
risk_df.rename(columns={'count':'Total Txn', 'sum':'Fraud Txn'}, inplace=True)
risk_df['Fraud Rate (%)'] = (risk_df['Fraud Txn'] / risk_df['Total Txn'] * 100).round(2)
st.dataframe(risk_df.sort_values('Fraud Rate (%)', ascending=False).head(10))

# Fraud vs Non-Fraud Transaction Map
st.subheader("🗺️ Fraud vs Non-Fraud Transaction Map")
map_sample = data.sample(1000, random_state=1)
map_sample.rename(columns={'long': 'lon'}, inplace=True)
st.map(map_sample[['lat', 'lon']])
fraud_map = map_sample[map_sample['is_fraud'] == 1]
st.map(fraud_map[['lat', 'lon']])

# Fraud by Hour
st.subheader("⏰ Fraud Occurrence by Hour of Day")
data['hour'] = pd.to_datetime(data['unix_time'], unit='s').dt.hour
fraud_by_hour = data[data['is_fraud']==1]['hour'].value_counts().sort_index()
fig_hour, ax_hour = plt.subplots()
ax_hour.bar(fraud_by_hour.index, fraud_by_hour.values)
ax_hour.set_xlabel("Hour of Day")
ax_hour.set_ylabel("Number of Frauds")
st.pyplot(fig_hour)

# Fraud by Day
st.subheader("📆 Fraud Occurrence by Day of Week")
data['day'] = pd.to_datetime(data['unix_time'], unit='s').dt.day_name()
fraud_by_day = data[data['is_fraud']==1]['day'].value_counts()[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]
fig_day, ax_day = plt.subplots()
ax_day.bar(fraud_by_day.index, fraud_by_day.values)
ax_day.set_xlabel("Day of Week")
ax_day.set_ylabel("Number of Frauds")
st.pyplot(fig_day)

# --- New Section: Predicted Fraud by Hour ---
st.subheader("📈 Predicted Fraud by Hour of Day")
pred_hour = pd.to_datetime(data.iloc[idx_test]['unix_time'], unit='s').dt.hour
pred_hour_df = pd.DataFrame({'hour': pred_hour, 'y_pred': y_pred})
pred_hour_counts = pred_hour_df[pred_hour_df['y_pred'] == 1]['hour'].value_counts().sort_index()
fig_pred_hour, ax_pred_hour = plt.subplots()
ax_pred_hour.bar(pred_hour_counts.index, pred_hour_counts.values)
ax_pred_hour.set_xlabel("Hour of Day")
ax_pred_hour.set_ylabel("# Predicted Frauds")
st.pyplot(fig_pred_hour)

# --- New Section: Predicted Fraud by Day of Week ---
st.subheader("📆 Predicted Fraud by Day of Week")
pred_day = pd.to_datetime(data.iloc[idx_test]['unix_time'], unit='s').dt.day_name()
pred_day_df = pd.DataFrame({'day': pred_day, 'y_pred': y_pred})
pred_day_counts = pred_day_df[pred_day_df['y_pred'] == 1]['day'].value_counts()
pred_day_counts = pred_day_counts.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
fig_pred_day, ax_pred_day = plt.subplots()
ax_pred_day.bar(pred_day_counts.index, pred_day_counts.values)
ax_pred_day.set_xlabel("Day of Week")
ax_pred_day.set_ylabel("# Predicted Frauds")
st.pyplot(fig_pred_day)

# --- New Section: Predicted Fraud Map ---
st.subheader("🗺️ Predicted Fraud Map")
pred_map_df = data.iloc[idx_test].copy()
pred_map_df['y_pred'] = y_pred
pred_map_df = pred_map_df.rename(columns={'long': 'lon'})
fraud_preds = pred_map_df[pred_map_df['y_pred'] == 1]
st.map(fraud_preds[['lat', 'lon']])

# --- New Section: Predicted Fraud Rate per Card ---
st.subheader("📊 Top Cards by Predicted Fraud Rate")
pred_card_df = pd.DataFrame({'cc_num': data.iloc[idx_test]['cc_num'], 'y_pred': y_pred})
pred_card_risk = pred_card_df.groupby('cc_num')['y_pred'].agg(['count','sum'])
pred_card_risk['Predicted Fraud Rate (%)'] = (pred_card_risk['sum'] / pred_card_risk['count'] * 100).round(2)
st.dataframe(pred_card_risk.sort_values('Predicted Fraud Rate (%)', ascending=False).head(10))

# --- New Section: Predicted Timeline Explorer ---
st.subheader("🔎 Predicted Timeline Explorer")
selected_pred_card = st.selectbox("Select Card Number (Predicted)", pred_card_df['cc_num'].unique())
if selected_pred_card:
    selected_df = data.iloc[idx_test].copy()
    selected_df['y_pred'] = y_pred
    card_txn = selected_df[selected_df['cc_num'] == selected_pred_card].sort_values('unix_time')
    fig_pred_timeline, ax_pred_timeline = plt.subplots(figsize=(10, 2))
    ax_pred_timeline.plot(card_txn['unix_time'], card_txn['amt'], marker='o', label='Amount', color='blue')
    fraud_txn = card_txn[card_txn['y_pred'] == 1]
    ax_pred_timeline.scatter(fraud_txn['unix_time'], fraud_txn['amt'], color='orange', label='Predicted Fraud', s=80)
    ax_pred_timeline.set_xlabel("Unix Time")
    ax_pred_timeline.set_ylabel("Amount")
    ax_pred_timeline.set_title(f"Predicted Timeline for Card: {selected_pred_card}")
    ax_pred_timeline.legend()
    st.pyplot(fig_pred_timeline)
    st.dataframe(card_txn[['unix_time', 'amt', 'y_pred', 'merchant', 'city', 'state', 'zip']])

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
