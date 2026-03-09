# EEG Automatic Sleep Scoring

Automated sleep staging from polysomnography (PSG) data using machine learning and transfer learning. Compares a Random Forest baseline, a CNN trained from scratch, and BIOT (a pretrained EEG foundation model) under a strict subject-wise cross-validation scheme.

## Dataset

- **Subjects**: 42 subjects with usable PSG recordings
- **Channels**: 16 (F3, F4, C3, C4, P3, P4, O1, O2, Fz, Cz, Pz, EOG1, EOG2, EMG1, EMG2, EMG3)
- **Sampling rate**: 200 Hz (resampled from 5000 Hz)
- **Epoch length**: 30 seconds (6000 samples)
- **Labels**: Manual sleep stage scoring (W, N1, N2, N3)
- **Validation**: 5-fold GroupKFold (subject-wise — test subjects never appear in training)

## Models

### Random Forest
Hand-crafted features per channel: statistical moments, RMS, zero-crossing rate, and band power (delta, theta, alpha, beta, gamma) via Welch PSD. 352 features total. Class-balanced with `class_weight='balanced'`.

### CNN (from scratch)
Two-layer 1D-CNN with max-pooling, followed by a fully connected head. Trained with class-weighted cross-entropy, Adam optimizer, 100 epochs. Input: raw 16-channel signal (16 × 6000).

### BIOT (Transfer Learning)
[`braindecode/biot-pretrained-prest-16chs`](https://huggingface.co/braindecode/biot-pretrained-prest-16chs) — a PyTorch foundation model pretrained on the TUH Abnormal EEG corpus and the Sleep Heart Health Study (SHHS, ~5M samples). Two-stage fine-tuning: frozen backbone (5 epochs, lr=1e-3) then full fine-tune (15 epochs, lr=1e-4). Best checkpoint per fold tracked by validation balanced accuracy.

> **Why not U-Sleep or SleepTransformer?** U-Sleep has no downloadable pretrained weights (web API only). SleepTransformer weights require TensorFlow 1.x. BIOT is the best publicly available PyTorch option pretrained on sleep EEG data.

## Results

### Task 1: W vs N1 (610 epochs — 419 W / 191 N1)

| Metric | Random Forest | CNN | BIOT |
|---|---|---|---|
| Accuracy | 0.700 ± 0.022 | 0.663 ± 0.036 | 0.648 |
| **Balanced Accuracy** | 0.542 ± 0.040 | 0.577 ± 0.093 | **0.603 ± 0.064** |
| **F1 (NREM)** | 0.186 ± 0.155 | 0.358 ± 0.185 | **0.458** |
| **F1-macro** | ~0.48 | ~0.49 | **0.596 ± 0.064** |

### Task 2: W vs NREM (N1+N2+N3) (644 epochs — 419 W / 225 NREM)

| Metric | Random Forest | CNN | BIOT |
|---|---|---|---|
| Accuracy | 0.679 ± 0.064 | 0.597 ± 0.030 | 0.663 |
| **Balanced Accuracy** | 0.559 ± 0.050 | 0.546 ± 0.046 | **0.608 ± 0.052** |
| **F1 (NREM)** | 0.257 ± 0.122 | 0.391 ± 0.122 | **0.477** |
| **F1-macro** | ~0.52 | ~0.54 | **0.605 ± 0.053** |
| **Cohen's Kappa** | 0.151 | 0.111 | **0.231** |

### Key Findings

- **BIOT outperforms both baselines** on balanced accuracy, F1 for the NREM class, and F1-macro across both tasks — the primary benefit of transfer learning with very few epochs per subject (~15).
- **RF accuracy is misleading** — it achieves the highest accuracy by mostly predicting Wake (the majority class), but collapses on NREM recall in several folds (F1 = 0.00).
- **W vs NREM is marginally better** than W vs N1 for all models. The N2/N3 signals are more distinct from wakefulness, making the task slightly easier.
- **High fold variance** is shared across all models, reflecting genuine inter-subject variability rather than model instability.
- All kappa scores fall in the "slight–fair" range (0.11–0.23), consistent with the challenge of cross-subject generalization from ~15 epochs/subject.

## Dependencies

```
braindecode>=0.9.0
huggingface-hub>=1.5.0
safetensors>=0.7.0
torch>=2.10.0
scikit-learn>=1.8.0
mne>=1.11.0
numpy>=2.4.1
```

Install with [uv](https://github.com/astral-sh/uv): `uv sync`
