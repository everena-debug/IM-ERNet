# IM-ERNet: A Multi-Task AST-BiLSTM Framework for Indian Classical Music Instrument and Emotion Classification

This repository contains the code, configuration, random seeds, per-run predictions and result tables for the paper
"IM-ERNet: A Multi-Task AST-BiLSTM Framework for Indian Classical Music Instrument and Emotion Classification".

IM-ERNet encodes a 10 s log-Mel spectrogram with an AudioSet-pretrained Audio Spectrogram Transformer (AST),
regroups the 12 × 101 patch tokens and averages them over frequency to obtain 101 time steps, models that sequence
with a BiLSTM (384 units per direction), forms a clip vector by attention pooling and predicts the instrument and the
emotion with two softmax heads. Training has two stages: the AST is frozen for 5 epochs, then its last four blocks are
fine-tuned for up to 15 epochs with discriminative learning rates.

## Repository layout

```
IM-ERNet/
├── code/
│   ├── imernet.py                      # complete pipeline (data audit, training, baselines, ablations, analysis)
│   └── IM-ERNet_experiments.ipynb      # the same pipeline as a Kaggle/Colab notebook
├── data/
│   ├── file_list.csv                   # every clip: instrument, emotion and duplicate group
│   └── removed_identical_copies.csv    # the 23 byte-identical copies removed before splitting
├── results/
│   ├── tables/                         # all CSV tables used in the paper
│   ├── figures/                        # result figures
│   ├── preds.zip                       # test and validation logits of every run (.npz), unzip to results/preds/
│   ├── logs.zip                        # epoch-wise training history of every run, unzip to results/logs/
│   └── results.jsonl                   # one line of metrics per run
├── config.json                         # full configuration used for the paper
├── requirements.txt
└── LICENSE
```

## Environment used in the paper

| Item | Value |
|---|---|
| GPU | NVIDIA Tesla T4, 16 GB (15.6 GB usable) |
| CPU / RAM | Intel Xeon 2.00 GHz, 4 cores / 33.7 GB |
| Platform | Kaggle notebook, Linux |
| Software | Python 3.13, PyTorch 2.11 (CUDA 13.0), Transformers 5.20, librosa 0.11, scikit-learn 1.6.1 |
| Run time | 35 runs in about 2.5 GPU hours |

Any CUDA GPU with at least 12 GB of memory should work with batch size 8 and mixed precision.

```bash
pip install -r requirements.txt
```

## Data

Folder layout expected by the code:

```
<DATA_DIR>/<Instrument>/<Emotion>/<clip>.wav | .mp3
Instruments: Veena, Tabla, Violin, Flute
Emotions:    Happy, Sad, Calm, Relax
```

The dataset has 753 clips; 23 byte-identical copies are removed automatically, leaving 730 clips.
`data/file_list.csv` lists every clip with its labels and its near-duplicate group (589 groups, cosine similarity
of log-Mel statistics ≥ 0.998). The audio clips are available from the corresponding author on reasonable request.

Emotion labels come from the mood stated for each source recording. For Tabla, the label coincides with the tala
(Trital = Happy, Jhaptal = Sad, Rupak = Calm).

## Reproducing the results

1. Place the dataset in the folder layout above.
2. Open `code/imernet.py` (or the notebook) and set `CFG["DATA_DIR"]` to the dataset root and `CFG["OUT_DIR"]` to an output folder.
3. Run:

```bash
python code/imernet.py
```

On Kaggle, upload the notebook, attach the dataset, select a GPU accelerator and choose "Save Version > Save & Run All".

The script performs, in order: duplicate audit, group-aware stratified splitting, IM-ERNet training (seeds 42, 52, 62, 72, 82),
the three baselines (AST, CNN-BiLSTM-Attn, HTS-AT; five seeds each), five ablations (seeds 42, 52, 62), statistical tests,
complexity measurement, noise robustness, calibration and failure analysis. All tables and figures are written to `OUT_DIR`.
If a run is interrupted, set `CFG["RESUME_DIR"]` to the previous output folder; finished runs are skipped.

## Protocol summary

| Setting | Value |
|---|---|
| Input | 16 kHz mono, silence trim (30 dB), peak normalization, 10 s crop or zero-pad |
| Features | 128-bin Kaldi log-Mel, 25 ms window, 10 ms hop, padded to 1024 × 128 |
| Encoder | MIT/ast-finetuned-audioset-10-10-0.4593 (12 layers, 12 heads, D = 768) |
| Temporal model | 1-layer BiLSTM, 384 units per direction |
| Loss | label-smoothed cross-entropy (0.1); 1.5 × emotion + 1.0 × instrument |
| Stage 1 | AST frozen, 5 epochs, AdamW lr 1e-3 |
| Stage 2 | last 4 AST blocks unfrozen, up to 15 epochs, lr 2e-5 (AST) / 1e-4 (heads), early stopping patience 5 |
| Other | weight decay 1e-2, ReduceLROnPlateau (0.5, patience 2), gradient clipping 1.0, SpecAugment p = 0.5, batch 8 |
| Split | StratifiedGroupKFold over 16 instrument-emotion strata, about 70/10/20; a new split per seed |
| Seeds | 42, 52, 62, 72, 82 (each seed fixes the split and the initialization) |

## Main results (test set, mean ± SD over five seeds)

| Model | Emotion accuracy (%) | Instrument accuracy (%) | Parameters (M) |
|---|---|---|---|
| AST | 93.60 ± 0.99 | 99.57 ± 0.66 | 86.79 |
| CNN-BiLSTM-Attn | 94.47 ± 0.88 | 97.71 ± 1.59 | 1.28 |
| HTS-AT | 95.02 ± 0.95 | 100.00 ± 0.00 | 28.13 |
| IM-ERNet | 94.46 ± 0.99 | 99.71 ± 0.39 | 90.33 |

Statistical tests are in `results/tables/significance_emo.csv` and `significance_inst.csv`.

## Reading the saved predictions

```python
import numpy as np
# first unzip results/preds.zip into results/preds/
d = np.load("results/preds/IM_ERNet_seed42.npz")
# idx: test indices into the de-duplicated file list; y_emo, y_inst: true labels
# logits_emo, logits_inst: test logits; val_*: the same for the validation set
acc = (d["logits_emo"].argmax(1) == d["y_emo"]).mean()
```

## Citation

If you use this code, please cite the paper (details will be added after publication).

## License

The code is released under the MIT License (see LICENSE). Audio clips from public sources remain under their original licences.
