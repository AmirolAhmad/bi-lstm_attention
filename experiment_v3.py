"""Leakage-controlled credit-card fraud experiment for reviewer verification.

The script uses chronological partitions, fits preprocessing on training data only,
constructs card-level windows independently inside each partition, and tunes the
classification threshold on validation data before one-time test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sklearn
import tensorflow as tf
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
DATASET_NAME = "dazzle-nu/CIS435-CreditCardFraudDetection"
FEATURES = [
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "distance_km",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_v3")
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def set_reproducibility() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_and_engineer(max_rows: int | None) -> pd.DataFrame:
    frame = load_dataset(DATASET_NAME, split="train").to_pandas()
    required = {"cc_num", "unix_time", "is_fraud", "amt", "city_pop", "lat", "long", "merch_lat", "merch_long"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = frame.sort_values(["unix_time", "cc_num"], kind="mergesort").reset_index(drop=True)
    if max_rows is not None and max_rows < len(frame):
        # A prefix preserves chronology for a reproducible smoke test. The paper run
        # omits --max-rows and therefore uses the complete dataset.
        frame = frame.iloc[:max_rows].copy()

    dt = pd.to_datetime(frame["unix_time"], unit="s", utc=True)
    hour = dt.dt.hour.to_numpy(dtype=np.float32)
    weekday = dt.dt.dayofweek.to_numpy(dtype=np.float32)
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)

    lat1 = np.radians(frame["lat"].to_numpy(dtype=np.float64))
    lon1 = np.radians(frame["long"].to_numpy(dtype=np.float64))
    lat2 = np.radians(frame["merch_lat"].to_numpy(dtype=np.float64))
    lon2 = np.radians(frame["merch_long"].to_numpy(dtype=np.float64))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    frame["distance_km"] = (6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float32)
    frame["cc_num"] = frame["cc_num"].astype(str)
    frame["is_fraud"] = frame["is_fraud"].astype(np.int8)
    return frame


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    times = frame["unix_time"].to_numpy()
    train_cut = times[int(0.70 * len(times))]
    val_cut = times[int(0.85 * len(times))]
    train = frame[frame["unix_time"] < train_cut].copy()
    val = frame[(frame["unix_time"] >= train_cut) & (frame["unix_time"] < val_cut)].copy()
    test = frame[frame["unix_time"] >= val_cut].copy()
    split_meta = {
        "train_cutoff_utc": pd.to_datetime(train_cut, unit="s", utc=True).isoformat(),
        "validation_cutoff_utc": pd.to_datetime(val_cut, unit="s", utc=True).isoformat(),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "fraud_rows": {
            "train": int(train["is_fraud"].sum()),
            "validation": int(val["is_fraud"].sum()),
            "test": int(test["is_fraud"].sum()),
        },
    }
    return train, val, test, split_meta


def scale_partitions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    scaler = StandardScaler()
    train.loc[:, FEATURES] = scaler.fit_transform(train[FEATURES]).astype(np.float32)
    val.loc[:, FEATURES] = scaler.transform(val[FEATURES]).astype(np.float32)
    test.loc[:, FEATURES] = scaler.transform(test[FEATURES]).astype(np.float32)
    return train, val, test, scaler


def make_sequences(frame: pd.DataFrame, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    short_cards = 0
    for _, group in frame.sort_values(["cc_num", "unix_time"], kind="mergesort").groupby("cc_num", sort=False):
        values = group[FEATURES].to_numpy(dtype=np.float32, copy=True)
        target = group["is_fraud"].to_numpy(dtype=np.int8, copy=True)
        if len(group) < length:
            short_cards += 1
            continue
        view = np.lib.stride_tricks.sliding_window_view(values, length, axis=0)
        windows.append(np.transpose(view, (0, 2, 1)).copy())
        labels.append(target[length - 1 :].copy())
    if not windows:
        raise ValueError("No sequences were created; reduce sequence length or check the data.")
    x = np.concatenate(windows, axis=0)
    y = np.concatenate(labels, axis=0)
    meta = {
        "sequences": int(len(y)),
        "fraud_sequences": int(y.sum()),
        "legitimate_sequences": int(len(y) - y.sum()),
        "cards_with_fewer_than_sequence_length": int(short_cards),
    }
    return x, y, meta


def build_model(kind: str, sequence_length: int, feature_count: int) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length, feature_count), name="transactions")
    if kind == "lstm":
        x = layers.LSTM(64, return_sequences=False, name="lstm")(inputs)
    elif kind == "bilstm":
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="bilstm")(inputs)
    elif kind == "bilstm_attention":
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_1")(inputs)
        x = layers.Dropout(0.2, name="dropout_1")(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name="bilstm_2")(x)
        scores = layers.Dense(1, activation="tanh", name="attention_scores")(x)
        weights = layers.Softmax(axis=1, name="attention_weights")(scores)
        weighted = layers.Multiply(name="weighted_states")([x, weights])
        x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1), name="context_vector")(weighted)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    x = layers.Dropout(0.2, name="dropout_output")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_probability")(x)
    model = keras.Model(inputs, outputs, name=kind)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="ROC", name="auroc"), keras.metrics.AUC(curve="PR", name="auprc")],
    )
    return model


def choose_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_metrics(name: str, y_true: np.ndarray, score: np.ndarray, threshold: float, train_seconds: float, latency: dict) -> dict:
    pred = (score >= threshold).astype(np.int8)
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auroc": roc_auc_score(y_true, score),
        "auprc": average_precision_score(y_true, score),
        "training_seconds": train_seconds,
        "inference_ms_per_sequence_median": latency["median_ms"],
        "inference_ms_per_sequence_p95_batch": latency["p95_ms"],
        "tn": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 1]),
    }


def keras_predict_with_latency(model: keras.Model, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, dict]:
    sample = x[: min(len(x), 10_000)]
    _ = model.predict(sample[: min(len(sample), batch_size)], batch_size=batch_size, verbose=0)
    timings = []
    chunks = []
    for start in range(0, len(sample), batch_size):
        batch = sample[start : start + batch_size]
        begin = time.perf_counter()
        chunks.append(model.predict(batch, batch_size=batch_size, verbose=0).reshape(-1))
        timings.append((time.perf_counter() - begin) * 1000.0 / len(batch))
    score = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1)
    return score, {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))}


def sklearn_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = np.clip(model.decision_function(x), -30, 30)
    return 1.0 / (1.0 + np.exp(-decision))


def sklearn_predict_with_latency(model, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, dict]:
    sample = x[: min(len(x), 10_000)]
    timings = []
    for start in range(0, len(sample), batch_size):
        batch = sample[start : start + batch_size]
        begin = time.perf_counter()
        _ = sklearn_score(model, batch)
        timings.append((time.perf_counter() - begin) * 1000.0 / len(batch))
    return sklearn_score(model, x), {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))}


def plot_results(output: Path, metrics: pd.DataFrame, curves: dict, histories: dict, y_test: np.ndarray) -> None:
    display = metrics.set_index("model")[["precision", "recall", "f1", "auroc", "auprc"]] * 100
    ax = display.plot(kind="bar", figsize=(11, 5), ylim=(0, 101), width=0.82)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, fontsize=8, loc="lower center")
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=240)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, score in curves.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        p, r, _ = precision_recall_curve(y_test, score)
        row = metrics[metrics["model"] == name].iloc[0]
        ax1.plot(fpr, tpr, label=f"{name} ({row.auroc:.3f})")
        ax2.plot(r, p, label=f"{name} ({row.auprc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax1.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    ax2.axhline(float(np.mean(y_test)), linestyle="--", color="grey", linewidth=0.8)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "roc_pr_curves.png", dpi=240)
    plt.close()

    for name, history in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].plot(history["loss"], label="train")
        axes[0].plot(history["val_loss"], label="validation")
        axes[0].set(title=f"{name}: loss", xlabel="Epoch", ylabel="Binary cross-entropy")
        axes[1].plot(history["auprc"], label="train")
        axes[1].plot(history["val_auprc"], label="validation")
        axes[1].set(title=f"{name}: AUPRC", xlabel="Epoch", ylabel="AUPRC")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend()
        plt.tight_layout()
        plt.savefig(output / f"history_{name}.png", dpi=220)
        plt.close()


def main() -> None:
    args = parse_args()
    set_reproducibility()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_and_engineer(args.max_rows)
    train, val, test, split_meta = chronological_split(frame)
    train, val, test, scaler = scale_partitions(train, val, test)
    x_train, y_train, train_meta = make_sequences(train, args.sequence_length)
    x_val, y_val, val_meta = make_sequences(val, args.sequence_length)
    x_test, y_test, test_meta = make_sequences(test, args.sequence_length)
    np.savez(output / "scaler_parameters.npz", mean=scaler.mean_, scale=scaler.scale_, features=np.array(FEATURES))

    negatives, positives = np.bincount(y_train, minlength=2)
    class_weight = {0: 1.0, 1: float(negatives / max(positives, 1))}
    metrics_rows: list[dict] = []
    curves: dict[str, np.ndarray] = {}
    histories: dict[str, dict] = {}

    flat_train = x_train.reshape(len(x_train), -1)
    flat_val = x_val.reshape(len(x_val), -1)
    flat_test = x_test.reshape(len(x_test), -1)
    classical = {
        "Decision Tree": DecisionTreeClassifier(max_depth=16, min_samples_leaf=5, class_weight="balanced", random_state=SEED),
        "Linear SVM": SGDClassifier(loss="hinge", class_weight="balanced", max_iter=1500, tol=1e-4, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    }
    for name, model in classical.items():
        begin = time.perf_counter()
        model.fit(flat_train, y_train)
        train_seconds = time.perf_counter() - begin
        val_score = sklearn_score(model, flat_val)
        threshold = choose_threshold(y_val, val_score)
        test_score, latency = sklearn_predict_with_latency(model, flat_test, args.batch_size)
        metrics_rows.append(compute_metrics(name, y_test, test_score, threshold, train_seconds, latency))
        curves[name] = test_score

    for kind, name in [("lstm", "LSTM"), ("bilstm", "BiLSTM"), ("bilstm_attention", "BiLSTM + Attention")]:
        tf.keras.backend.clear_session()
        set_reproducibility()
        model = build_model(kind, args.sequence_length, len(FEATURES))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_auprc", mode="max", factor=0.5, patience=2, min_lr=1e-5),
        ]
        begin = time.perf_counter()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            shuffle=False,
            callbacks=callbacks,
            verbose=2,
        )
        train_seconds = time.perf_counter() - begin
        histories[name] = {key: [float(v) for v in values] for key, values in history.history.items()}
        val_score = model.predict(x_val, batch_size=args.batch_size, verbose=0).reshape(-1)
        threshold = choose_threshold(y_val, val_score)
        test_score, latency = keras_predict_with_latency(model, x_test, args.batch_size)
        metrics_rows.append(compute_metrics(name, y_test, test_score, threshold, train_seconds, latency))
        curves[name] = test_score
        model.save(output / f"{kind}.keras")

        if kind == "bilstm_attention":
            attention_model = keras.Model(model.input, model.get_layer("attention_weights").output)
            fraud_idx = np.flatnonzero(y_test == 1)
            example_idx = int(fraud_idx[0]) if len(fraud_idx) else 0
            weights = attention_model.predict(x_test[example_idx : example_idx + 1], verbose=0).reshape(-1)
            pd.DataFrame({"step": np.arange(1, args.sequence_length + 1), "attention_weight": weights}).to_csv(
                output / "attention_example.csv", index=False
            )
            plt.figure(figsize=(7, 3.5))
            plt.bar(np.arange(1, args.sequence_length + 1), weights)
            plt.xlabel("Transaction position in sequence")
            plt.ylabel("Learned attention weight")
            plt.title("Attention weights for one fraudulent test sequence")
            plt.xticks(np.arange(1, args.sequence_length + 1))
            plt.tight_layout()
            plt.savefig(output / "attention_example.png", dpi=240)
            plt.close()

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    plot_results(output, metrics, curves, histories, y_test)

    metadata = {
        "dataset": DATASET_NAME,
        "dataset_rows_used": len(frame),
        "full_dataset_run": args.max_rows is None,
        "seed": SEED,
        "sequence_length": args.sequence_length,
        "sequence_stride": 1,
        "sequence_label": "is_fraud of the final transaction",
        "split_policy": "70/15/15 chronological; sequences rebuilt inside each partition",
        "features": FEATURES,
        "categorical_features": [],
        "imbalance_method": "training-only class weights; no SMOTE",
        "class_weight": class_weight,
        "split": split_meta,
        "sequence_counts": {"train": train_meta, "validation": val_meta, "test": test_meta},
        "software": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
