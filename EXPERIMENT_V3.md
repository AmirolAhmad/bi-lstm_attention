# Leakage-controlled reviewer experiment

This experiment is separate from the deployed Streamlit application. It is used
to produce auditable numbers for the revised paper.

Key controls:

- complete CIS435/Sparkov dataset (unless `--max-rows` is explicitly supplied);
- 70/15/15 chronological transaction split;
- scaler fitted on training rows only;
- per-card, length-10, non-overlapping stride-10 windows rebuilt independently
  in each split;
- no sequence crosses a split boundary;
- class weighting from training labels only; SMOTE is not used;
- validation-only threshold selection, followed by one-time test evaluation;
- trainable temporal attention whose weights are taken from the fitted model;
- fixed seed, early stopping, software/hardware metadata, training time and
  inference latency;
- Decision Tree, linear SVM, Random Forest, LSTM and BiLSTM comparisons.

The workflow uploads `results_v3/` as an Actions artifact. The folder contains
`metrics.csv`, `run_metadata.json`, plots, learned models, scaler parameters and
an attention example.

Run locally:

```bash
python -m pip install -r requirements-experiment.txt
python experiment_v3.py --output-dir results_v3
```

For a smoke test only:

```bash
python experiment_v3.py --max-rows 100000 --epochs 1
```
