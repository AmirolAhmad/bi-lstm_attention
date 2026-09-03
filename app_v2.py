"""Streamlit runner for the leakage-controlled paper experiment."""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from experiment_v3 import (
    DATASET_NAME,
    FEATURES,
    build_model,
    choose_threshold,
    chronological_split,
    compute_metrics,
    keras_predict_with_latency,
    load_and_engineer,
    make_sequences,
    scale_partitions,
    set_reproducibility,
)


st.set_page_config(page_title="Leakage-Controlled BiLSTM Attention", layout="wide")
st.title("Credit Card Fraud Detection: Corrected Experiment")
st.caption("Chronological split • training-only scaling • card-level sequences • learned temporal attention")

with st.sidebar:
    st.header("Experiment settings")
    run_scope = st.selectbox("Data scope", ["Full dataset (paper run)", "First 200,000 rows (smoke test)"])
    sequence_length = st.number_input("Sequence length", min_value=2, max_value=50, value=10, step=1)
    sequence_stride = st.number_input("Sequence stride", min_value=1, max_value=50, value=10, step=1)
    epochs = st.number_input("Maximum epochs", min_value=1, max_value=30, value=12, step=1)
    batch_size = st.selectbox("Batch size", [128, 256, 512], index=2)
    start = st.button("Run corrected experiment", type="primary", use_container_width=True)

st.info(
    "This version removes the leakage in the earlier app. Transactions are divided by time before "
    "scaling or sequence construction. SMOTE is not used; imbalance is handled with class weights "
    "calculated from training sequences only."
)

if not start:
    st.write("Select the paper run and press **Run corrected experiment**. The full run may take several minutes.")
    st.stop()

set_reproducibility()
max_rows = None if run_scope.startswith("Full") else 200_000

progress = st.progress(0, text="Loading and preparing the dataset...")
frame = load_and_engineer(max_rows)
progress.progress(15, text="Creating chronological partitions...")
train, val, test, split_meta = chronological_split(frame)
train, val, test, _ = scale_partitions(train, val, test)
progress.progress(30, text="Constructing card-level sequences inside each partition...")
x_train, y_train, train_meta = make_sequences(train, int(sequence_length), int(sequence_stride))
x_val, y_val, val_meta = make_sequences(val, int(sequence_length), int(sequence_stride))
x_test, y_test, test_meta = make_sequences(test, int(sequence_length), int(sequence_stride))

negative, positive = np.bincount(y_train, minlength=2)
class_weight = {0: 1.0, 1: float(negative / max(positive, 1))}

progress.progress(40, text="Training BiLSTM with learned attention...")
model = build_model("bilstm_attention", int(sequence_length), len(FEATURES))
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_auprc", mode="max", factor=0.5, patience=2, min_lr=1e-5),
]
begin = time.perf_counter()
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=int(epochs),
    batch_size=int(batch_size),
    class_weight=class_weight,
    shuffle=False,
    callbacks=callbacks,
    verbose=0,
)
training_seconds = time.perf_counter() - begin

progress.progress(85, text="Selecting the threshold on validation data...")
validation_score = model.predict(x_val, batch_size=int(batch_size), verbose=0).reshape(-1)
threshold = choose_threshold(y_val, validation_score)
test_score, latency = keras_predict_with_latency(model, x_test, int(batch_size))
result = compute_metrics(
    "BiLSTM + Attention", y_test, test_score, threshold, training_seconds, latency
)
progress.progress(100, text="Experiment complete")

st.subheader("Auditable data split")
split_table = pd.DataFrame(
    {
        "partition": ["Training", "Validation", "Test"],
        "rows": [split_meta["rows"]["train"], split_meta["rows"]["validation"], split_meta["rows"]["test"]],
        "fraud rows": [split_meta["fraud_rows"]["train"], split_meta["fraud_rows"]["validation"], split_meta["fraud_rows"]["test"]],
        "sequences": [train_meta["sequences"], val_meta["sequences"], test_meta["sequences"]],
        "fraud sequences": [train_meta["fraud_sequences"], val_meta["fraud_sequences"], test_meta["fraud_sequences"]],
    }
)
st.dataframe(split_table, hide_index=True, use_container_width=True)
st.caption(
    f"Training cutoff: {split_meta['train_cutoff_utc']} • Validation cutoff: "
    f"{split_meta['validation_cutoff_utc']} • Dataset: {DATASET_NAME}"
)

st.subheader("Held-out test performance")
cols = st.columns(6)
for col, label, key in zip(
    cols,
    ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"],
    ["accuracy", "precision", "recall", "f1", "auroc", "auprc"],
):
    col.metric(label, f"{100 * result[key]:.2f}%")

left, right = st.columns(2)
with left:
    st.write("Confusion matrix")
    matrix = confusion_matrix(y_test, test_score >= threshold, labels=[0, 1])
    st.dataframe(pd.DataFrame(matrix, index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"]))
    st.write(f"Validation-selected threshold: `{threshold:.6f}`")
    st.write(f"Training time: `{training_seconds:.2f} s`")
    st.write(f"Median inference latency: `{latency['median_ms']:.4f} ms/sequence`")

with right:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[1].plot(history.history["auprc"], label="train")
    axes[1].plot(history.history["val_auprc"], label="validation")
    axes[1].set(title="AUPRC", xlabel="Epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    st.pyplot(fig)

st.subheader("Learned attention weights")
fraud_indices = np.flatnonzero(y_test == 1)
example_index = int(fraud_indices[0]) if len(fraud_indices) else 0
attention_model = keras.Model(model.input, model.get_layer("attention_weights").output)
weights = attention_model.predict(x_test[example_index : example_index + 1], verbose=0).reshape(-1)
attention = pd.DataFrame({"Transaction position": np.arange(1, int(sequence_length) + 1), "Attention weight": weights})
st.bar_chart(attention.set_index("Transaction position"))
st.caption("Weights come from the attention layer of the fitted model, not from a separate untrained network.")

export = {
    "dataset": DATASET_NAME,
    "rows_used": len(frame),
    "full_dataset_run": max_rows is None,
    "features": FEATURES,
    "sequence_length": int(sequence_length),
    "sequence_stride": int(sequence_stride),
    "split": split_meta,
    "sequence_counts": {"train": train_meta, "validation": val_meta, "test": test_meta},
    "class_weight": class_weight,
    "epochs_completed": len(history.history["loss"]),
    "metrics": result,
    "tensorflow": tf.__version__,
}
st.download_button(
    "Download experiment JSON",
    data=json.dumps(export, indent=2),
    file_name="corrected_experiment_results.json",
    mime="application/json",
)
