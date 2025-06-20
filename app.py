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

st.title("💳 Credit Card Fraud Detection with Bi-LSTM + Attention (Sequence Input)")
st.markdown("---")
# --- Step 1: Load Data ---
@st.cache_data
def load_data():
    dataset = load_dataset("dazzle-nu/CIS435-CreditCardFraudDetection")
    df = dataset["train"].to_pandas()
    return df

with st.spinner("Loading data... Please wait"):
    data = load_data()
st.success("Data loaded!")
st.caption("Dataset contains credit card transactions with fraud labels from HuggingFace - dazzle-nu/CIS435-CreditCardFraudDetection.")
data = data.sample(n=50000, random_state=42).reset_index(drop=True)  # Reduce for demo speed

# --- Step 2: Prepare Sequential Data using cc_num ---
features = ['amt', 'city_pop', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long']
SEQ_LEN = 10  # Sequence length
data = data.sort_values(['cc_num', 'unix_time'])  # Important: Sort for proper sequencing

X = data[features].values
y = data['is_fraud'].values
cc_nums = data['cc_num'].values

# Normalize features before sequencing!
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
            y_seqs.append(card_y[i+SEQ_LEN-1])  # Label = last in window

X_seqs = np.array(X_seqs)
y_seqs = np.array(y_seqs)

st.write(f"Total sequences: {len(X_seqs)} (Sequence length: {SEQ_LEN})")

# --- Step 3: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_seqs, y_seqs, test_size=0.2, stratify=y_seqs, random_state=42
)

# --- Step 4: Attention Layer ---
class Attention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.return_attention = return_attention

    def call(self, inputs):
        score = K.softmax(K.sum(inputs, axis=-1), axis=-1)  # (batch, seq_len)
        score = K.expand_dims(score)
        attended = K.sum(inputs * score, axis=1)
        if self.return_attention:
            return attended, score
        return attended

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]
        else:
            return (input_shape[0], input_shape[-1])

# --- Step 5: Build Model ---
input_shape = (SEQ_LEN, len(features))
inputs = Input(shape=input_shape)
x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(32, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Attention()(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs, output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Attention model for visualization
inputs_att = Input(shape=input_shape)
x_att = Bidirectional(LSTM(64, return_sequences=True))(inputs_att)
x_att = Dropout(0.2)(x_att)
x_att = Bidirectional(LSTM(32, return_sequences=True))(x_att)
x_att = Dropout(0.2)(x_att)
att_layer = Attention(return_attention=True)
att_out, att_weights = att_layer(x_att)
output_att = Dense(1, activation='sigmoid')(att_out)
att_model = Model(inputs_att, [output_att, att_weights])

# --- Step 6: Train Model ---
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

with st.spinner("Training model... Please wait"):
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        class_weight=class_weights,
        verbose=0
    )
st.success("Training complete!")

# --- Step 7: Predict & Evaluate ---
with st.spinner("Predicting on test set..."):
    y_proba = model.predict(X_test).flatten()
y_pred = (y_proba > 0.5).astype(int)

st.subheader("📉 Training Loss & Accuracy")
fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 4))
ax_hist[0].plot(history.history['accuracy'], label='Train Acc')
ax_hist[0].plot(history.history['val_accuracy'], label='Val Acc')
ax_hist[0].set_title('Accuracy per Epoch')
ax_hist[0].legend()
ax_hist[1].plot(history.history['loss'], label='Train Loss')
ax_hist[1].plot(history.history['val_loss'], label='Val Loss')
ax_hist[1].set_title('Loss per Epoch')
ax_hist[1].legend()
st.pyplot(fig_hist)

st.subheader("📈 Model Evaluation (Classification Report)")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().round(2))

st.subheader("🟦 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm, cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
    ax_cm.text(j, i, str(z), ha='center', va='center')
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

st.subheader("📊 ROC Curve & AUC")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)

st.subheader("🔢 Precision, Recall, F1-score (Test Set)")
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
labels = ['Not Fraud', 'Fraud']
fig_bar, ax_bar = plt.subplots()
width = 0.2
x = np.arange(len(labels))
ax_bar.bar(x - width, prec, width=width, label='Precision')
ax_bar.bar(x, rec, width=width, label='Recall')
ax_bar.bar(x + width, f1, width=width, label='F1-score')
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

st.subheader("🔍 Attention Heatmap (Sample Fraud Sequence)")
# Find a fraud sample in test set
fraud_idxs = np.where((y_test == 1))[0]
if len(fraud_idxs) > 0:
    fraud_idx = fraud_idxs[0]
    fraud_sample = X_test[fraud_idx:fraud_idx+1]
    _, att_weight = att_model.predict(fraud_sample)
    att_weight = att_weight[0].flatten()
    fig_att, ax_att = plt.subplots()
    im = ax_att.imshow([att_weight], cmap='Reds', aspect='auto')
    ax_att.set_title("Attention Weights Across Sequence")
    ax_att.set_xlabel("Transaction Index in Sequence")
    fig_att.colorbar(im, ax=ax_att)
    st.pyplot(fig_att)
else:
    st.info("No fraud sample found in test set for attention visualization.")

st.subheader("🧭 Feature Importance (via Mean Absolute Value)")
# Proxy for feature importance per sequence step
feat_means = np.mean(np.abs(X_test), axis=(0,1))
fig_feat, ax_feat = plt.subplots()
ax_feat.bar(features, feat_means)
ax_feat.set_ylabel("Mean Absolute Value (Standardized)")
ax_feat.set_title("Feature Importance Proxy")
st.pyplot(fig_feat)

# Interactive Transaction Timeline (per card)
st.subheader("🔎 Card Transaction Timeline Explorer")
card_ids = data['cc_num'].unique()
selected_card = st.selectbox("Select cc_num (card)", card_ids)

if selected_card:
    with st.spinner(f"Loading transactions for {selected_card}..."):
        # Proses ambil data
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
    st.success("Card transaction data loaded!")

st.subheader("📊 Per-Card Fraud Risk Overview")
risk_df = data.groupby('cc_num')['is_fraud'].agg(['count', 'sum'])
risk_df.rename(columns={'count':'Total Txn', 'sum':'Fraud Txn'}, inplace=True)
risk_df['Fraud Rate (%)'] = (risk_df['Fraud Txn'] / risk_df['Total Txn'] * 100).round(2)
st.dataframe(risk_df.sort_values('Fraud Rate (%)', ascending=False).head(10))

# Per-card Fraud Risk
st.subheader("🗺️ Fraud vs Non-Fraud Transaction Map")
# Subsample for speed
map_sample = data.sample(1000, random_state=1)
map_sample.rename(columns={'long': 'lon'}, inplace=True)
st.map(map_sample[['lat', 'lon']])
fraud_map = map_sample[map_sample['is_fraud'] == 1]
st.map(fraud_map[['lat', 'lon']])
st.caption("Red dots = fraud (on second map)")

# Geo-Visualization (Map Plot)
st.subheader("⏰ Fraud Occurrence by Hour of Day")
data['hour'] = pd.to_datetime(data['unix_time'], unit='s').dt.hour
fraud_by_hour = data[data['is_fraud']==1]['hour'].value_counts().sort_index()
fig_hour, ax_hour = plt.subplots()
ax_hour.bar(fraud_by_hour.index, fraud_by_hour.values)
ax_hour.set_xlabel("Hour of Day")
ax_hour.set_ylabel("Number of Frauds")
st.pyplot(fig_hour)

# Time-based Analysis (Fraud by Hour/Day)
st.subheader("📆 Fraud Occurrence by Day of Week")
data['day'] = pd.to_datetime(data['unix_time'], unit='s').dt.day_name()
fraud_by_day = data[data['is_fraud']==1]['day'].value_counts()[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]
fig_day, ax_day = plt.subplots()
ax_day.bar(fraud_by_day.index, fraud_by_day.values)
ax_day.set_xlabel("Day of Week")
ax_day.set_ylabel("Number of Frauds")
st.pyplot(fig_day)


st.subheader("💾 Save 100 Sample Predictions to CSV")
sample_idx = np.random.choice(np.arange(X_test.shape[0]), 100, replace=False)
sample_data = X_test[sample_idx].reshape(100, SEQ_LEN * len(features))
sample_pred = y_pred[sample_idx]
sample_prob = y_proba[sample_idx]
sample_df = pd.DataFrame(sample_data, columns=[f"{f}_t{t+1}" for t in range(SEQ_LEN) for f in features])
sample_df['Predicted Fraud'] = sample_pred
sample_df['Fraud Probability'] = sample_prob

csv = sample_df.to_csv(index=False).encode()
st.download_button("Download CSV", data=csv, file_name="lstm_fraud_predictions.csv")

st.markdown("---")
st.caption("Model: Bi-LSTM + Attention | Data: dazzle-nu/CIS435-CreditCardFraudDetection | Author: Amirol Ahmad | License: MIT")
