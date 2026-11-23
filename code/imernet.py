# %% [markdown]
# # IM-ERNet v3: revision experiments for the IJITCS reviewer comments
#
# This notebook replaces the original training notebook. It runs every experiment the reviewers asked for on the
# Indian Classical Instrument Music Emotion dataset, under one protocol, and writes all tables and 300 dpi figures to `OUT_DIR`.
#
# | Reviewer comment | Section of this notebook |
# |---|---|
# | Major 1: multiple runs, mean ± SD, significance tests, leakage check | 4 (leakage), 5 (split), 9 (multi-seed runs), 10 (mean ± SD), 11 (t-test, Wilcoxon, McNemar, CI) |
# | Major 2: same-dataset baselines | 7 (AST, CNN-BiLSTM-Attn, Wav2Vec2, HTS-AT), 9, 10, 11 |
# | Major 3: ablation studies | 8 (registry), 9, 10, 11 |
# | Major 4: dataset transparency, stratified split, statistics | 3 (dataset statistics), 5 (split table), 15 (annotation agreement, Kappa) |
# | Major 6: complexity | 12 (parameters, FLOPs, latency, memory) |
# | Major 7: failure cases, noise robustness, playing style | 13 (acoustic error analysis, instrument × emotion), 14 (noise robustness), 16 (calibration) |
# | Major 8: reproducibility | 2 (environment), 17 (README, requirements, archive) |
# | Minor 11, 12, 14: hyperparameters, AST to BiLSTM interface, hardware | 2, 7, 12 |
#
# Corrections to the original notebook:
# 1. Labels are read from the folder structure (`<instrument>/<emotion>/file`). The original plotting code printed instrument
#    names in a different order from the label map (flute=0, tabla=1, veena=2, violin=3), so the instrument names in the old
#    tables and figures were wrong. Class names here come from one list.
# 2. The original notebook used an 80/20 train/validation split and reported the validation set, which also drove the
#    learning-rate scheduler and checkpoint saving (the saved checkpoint was never reloaded; the last epoch was reported). This notebook uses a stratified 70/10/20 train/validation/test split. The checkpoint
#    with the lowest validation loss is reloaded, and only the untouched test set is reported.
# 3. Every model is trained with 5 (or 3) different seeds. Each seed draws a new stratified split.
# 5. v3 changes, applied to every model under the same protocol: 10 s clips instead of 5 s (half of the AST input was
#    padding), label smoothing 0.1, weight decay 0.01, Stage-2 head learning rate 1e-4 (validation loss rose after epoch 6
#    in v2). In IM-ERNet the BiLSTM now reads the AST patch tokens in time order: the 12 x 101 frequency x time grid is
#    averaged over frequency, giving 101 steps of 100 ms.
# 6. Early stopping (patience 5 on validation loss) after Stage 1; the lowest-validation-loss checkpoint is still evaluated.
# 4. Byte-identical copies are removed before splitting (all copies when their folder labels disagree), and near-duplicate
#    clips are grouped so that they never fall in different subsets.
#
# Runtime: 35 runs (Wav2Vec2 and two ablations switched off, early stopping on); about 3.5 to 4 hours on a T4. Results are appended to `results.jsonl`, and finished
# runs are skipped. To continue in a new Kaggle session, save the output as a dataset, attach it, and set `RESUME_DIR`.

# %%
# Kaggle normally has these packages. Uncomment only if an import fails.
# !pip install -q "transformers>=4.40" librosa soundfile scikit-learn scipy psutil

# %% [markdown]
# ## 1. Configuration

# %%
import os, sys, gc, re, json, time, math, random, hashlib, platform, shutil, contextlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CFG = {
    "DATA_DIR": "/kaggle/input/datasets/prasannak642/indian-classical-instrument-music-emotion-dataset/Indian Classical Instrument Music Emotion Dataset",
    "OUT_DIR": "/kaggle/working/imernet_v3",
    "RESUME_DIR": "auto",          # "auto": continue from an attached earlier output (results.jsonl under /kaggle/input)
    "AST_NAME": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "W2V_NAME": "facebook/wav2vec2-base",
    "CLAP_NAME": "laion/clap-htsat-unfused",   # audio tower of this checkpoint is HTS-AT
    "SR": 16000,                   # AST, Wav2Vec2 and the CNN baseline use 16 kHz
    "CLAP_SR": 48000,              # HTS-AT (CLAP) expects 48 kHz
    "DURATION": 10.0,              # seconds per clip after trimming (fills the 10.24 s AST input window)
    "TOP_DB": 30,
    "SPLIT": (0.70, 0.10, 0.20),   # train / validation / test
    "DEDUP_IDENTICAL": True,       # drop byte-identical copies before splitting (all copies if their labels disagree)
    "GROUP_SPLIT": True,           # near-duplicate clips are grouped and kept in the same subset
    "MAIN_SEEDS": [42, 52, 62, 72, 82],
    "BASELINE_SEEDS": [42, 52, 62, 72, 82],
    "ABLATION_SEEDS": [42, 52, 62],
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 0,              # features are cached in memory, workers are not needed
    "EPOCHS_STAGE1": 5,
    "EPOCHS_STAGE2": 15,
    "LR_STAGE1": 1e-3,
    "LR_STAGE2_BACKBONE": 2e-5,
    "LR_STAGE2_HEAD": 1e-4,
    "LR_SCRATCH": 1e-3,
    "WEIGHT_DECAY": 1e-2,
    "LABEL_SMOOTHING": 0.1,
    "UNFREEZE_BLOCKS": 4,
    "W_EMO": 1.5,
    "W_INST": 1.0,
    "DROPOUT": 0.3,
    "GRAD_CLIP": 1.0,
    "EARLY_STOP_PATIENCE": 5,     # stop when validation loss has not improved for 5 epochs (after Stage 1)
    "SPECAUG": {"time_mask": 20, "freq_mask": 10, "p": 0.5},
    "RUN_MAIN": True,
    "RUN_BASELINES": True,
    "RUN_ABLATIONS": True,
    "RUN_W2V": False,              # weakest baseline in v2 (74%); off to save GPU time
    "RUN_HTSAT": True,
    "SAVE_CKPT_FOR": ["IM-ERNet", "AST [3]", "CNN-BiLSTM-Attn [6]"],   # used by noise robustness and embeddings
    "NOISE_TYPES": ["white", "pink"],
    "NOISE_SNRS": [30, 20, 10, 5, 0],
    "NEAR_DUP_THRESHOLD": 0.995,
    "ANNOTATION_CSV": None,        # CSV with columns: file, annotator_1, annotator_2, ... (emotion names)
}

INSTRUMENTS = ["Veena", "Tabla", "Violin", "Flute"]
EMOTIONS = ["Happy", "Sad", "Calm", "Relax"]

OUT = Path(CFG["OUT_DIR"])
for sub in ["figures", "tables", "preds", "ckpt", "logs", "cache"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
if CFG["RESUME_DIR"] == "auto":
    found = [p.parent for p in Path("/kaggle/input").rglob("results.jsonl") if p.parent.name == Path(CFG["OUT_DIR"]).name] \
        if Path("/kaggle/input").exists() else []
    CFG["RESUME_DIR"] = str(found[0]) if found else None
    print("Earlier output found:" if found else "No earlier output attached; starting fresh.", CFG["RESUME_DIR"] or "")
if CFG["RESUME_DIR"] and Path(CFG["RESUME_DIR"]).exists():
    shutil.copytree(CFG["RESUME_DIR"], OUT, dirs_exist_ok=True)
    print("Resumed from", CFG["RESUME_DIR"])
RESULTS_FILE = OUT / "results.jsonl"

# %% [markdown]
# ## 2. Imports, seeding and computing environment (Minor 14, Major 8)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import scipy
import sklearn
import transformers
from transformers import AutoFeatureExtractor, ASTModel, Wav2Vec2Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             cohen_kappa_score, matthews_corrcoef, confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "librosa": librosa.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}
    try:
        info["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l][0]
    except Exception:
        info["cpu_model"] = platform.processor() or "n/a"
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_memory_gb"] = round(p.total_memory / 1e9, 1)
    return info


ENV = env_info()
json.dump(ENV, open(OUT / "logs" / "environment.json", "w"), indent=2)
print(json.dumps(ENV, indent=2))

# %% [markdown]
# ## 3. Dataset indexing and statistics (Major 4)

# %%
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def canon(name, vocab):
    n = name.strip().lower()
    for v in vocab:
        if n == v.lower() or n.startswith(v.lower()):
            return v
    return None


rows, skipped = [], []
for root, _, files in os.walk(CFG["DATA_DIR"]):
    for f in sorted(files):
        if not f.lower().endswith(AUDIO_EXTS):
            continue
        p = Path(root) / f
        rel = p.relative_to(CFG["DATA_DIR"]).parts
        inst = canon(rel[0], INSTRUMENTS) if len(rel) >= 3 else None
        emo = canon(rel[1], EMOTIONS) if len(rel) >= 3 else None
        if inst is None or emo is None:
            skipped.append(str(p))
            continue
        rows.append({"path": str(p), "file": f, "instrument": inst, "emotion": emo})

df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
df["inst_id"] = df["instrument"].map({c: i for i, c in enumerate(INSTRUMENTS)})
df["emo_id"] = df["emotion"].map({c: i for i, c in enumerate(EMOTIONS)})
df["strata"] = df["inst_id"] * len(EMOTIONS) + df["emo_id"]


def make_group_id(path):
    # One group per file by default. If several clips were cut from one recording, return the recording name here
    # (for example re.sub(r"[_\- ]?\d+$", "", Path(path).stem)) and set GROUP_SPLIT=True.
    return Path(path).stem


df["group"] = df["path"].map(make_group_id)
Y_INST = df["inst_id"].values.astype(np.int64)
Y_EMO = df["emo_id"].values.astype(np.int64)
print(f"Usable clips: {len(df)} | skipped: {len(skipped)}")


def audio_info(p):
    try:
        i = sf.info(p)
        return float(i.duration), int(i.samplerate), int(i.channels)
    except Exception:
        y, sr = librosa.load(p, sr=None, mono=False)
        return y.shape[-1] / sr, sr, (1 if y.ndim == 1 else y.shape[0])


meta_file = OUT / "cache" / "metadata.csv"
if meta_file.exists() and len(pd.read_csv(meta_file)) == len(df):
    meta = pd.read_csv(meta_file)
    for c in ["duration_s", "orig_sr", "channels", "md5"]:
        df[c] = meta[c].values
else:
    info = [audio_info(p) for p in df["path"]]
    df["duration_s"], df["orig_sr"], df["channels"] = zip(*info)
    df["md5"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in df["path"]]
    df.to_csv(meta_file, index=False)

counts = pd.crosstab(df["instrument"], df["emotion"]).reindex(index=INSTRUMENTS, columns=EMOTIONS)
counts["Total"] = counts.sum(1)
counts.loc["Total"] = counts.sum(0)
counts.to_csv(OUT / "tables" / "dataset_counts.csv")
print(counts)

dur = df.groupby("instrument")["duration_s"].describe()[["count", "mean", "std", "min", "50%", "max"]].reindex(INSTRUMENTS)
dur.loc["All"] = df["duration_s"].describe()[["count", "mean", "std", "min", "50%", "max"]]
dur.round(2).to_csv(OUT / "tables" / "dataset_durations.csv")
print(dur.round(2))
fmt = df.assign(ext=df["file"].str.lower().str.extract(r"\.(\w+)$")[0]).groupby(["ext", "orig_sr", "channels"]).size()
fmt.to_csv(OUT / "tables" / "dataset_formats.csv")
print(fmt)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
counts.drop(index="Total", columns="Total").plot(kind="bar", ax=ax[0])
ax[0].set_ylabel("Number of clips"); ax[0].set_xlabel("Instrument"); ax[0].legend(title="Emotion")
for inst in INSTRUMENTS:
    ax[1].hist(df.loc[df.instrument == inst, "duration_s"], bins=30, alpha=0.6, label=inst)
ax[1].set_xlabel("Clip duration (s)"); ax[1].set_ylabel("Number of clips"); ax[1].legend()
savefig("dataset_distribution")

dups = df[df.duplicated("md5", keep=False)].sort_values("md5")
dups[["file", "instrument", "emotion", "md5"]].to_csv(OUT / "tables" / "exact_duplicates.csv", index=False)
conflict = dups.groupby("md5")["strata"].nunique()
conflict = set(conflict[conflict > 1].index)
print(f"Byte-identical files: {len(dups)} in {dups.md5.nunique()} sets | sets with conflicting labels: {len(conflict)}")
if CFG["DEDUP_IDENTICAL"]:
    keep = ~df["md5"].isin(conflict) & ~df.duplicated("md5", keep="first")
    removed = df[~keep]
    removed[["file", "instrument", "emotion", "md5"]].to_csv(OUT / "tables" / "removed_duplicates.csv", index=False)
    df = df[keep].reset_index(drop=True)
    Y_INST = df["inst_id"].values.astype(np.int64)
    Y_EMO = df["emo_id"].values.astype(np.int64)
    print(f"Removed {len(removed)} duplicate copies ({len(df)} clips remain)")
    clean_counts = pd.crosstab(df["instrument"], df["emotion"]).reindex(index=INSTRUMENTS, columns=EMOTIONS)
    clean_counts["Total"] = clean_counts.sum(1)
    clean_counts.loc["Total"] = clean_counts.sum(0)
    clean_counts.to_csv(OUT / "tables" / "dataset_counts_after_dedup.csv")
    print(clean_counts)

# %% [markdown]
# ## 4. Preprocessing, feature caching and near-duplicate detection (Major 1, Minor 12)
#
# Each clip is resampled, converted to mono, trimmed (top_db = 30), peak-normalised to [-1, 1], and cut or zero-padded to
# `DURATION` seconds. The AST feature extractor converts the waveform to 128 Kaldi log-Mel bins with a 25 ms window and
# 10 ms shift, and pads the result to 1024 frames. AST splits this map into 16 × 16 patches with stride 10
# (12 × 101 = 1212 patches), adds two class tokens, and returns 1214 tokens of size 768.
# IM-ERNet arranges the 1212 patch tokens on their 12 × 101 grid and averages over frequency, so the BiLSTM reads
# 101 time steps of 768 values in temporal order.

# %%
def load_wave(path, sr, duration=CFG["DURATION"], top_db=CFG["TOP_DB"]):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    y = y / (np.max(np.abs(y)) + 1e-8)
    n = int(sr * duration)
    y = np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]
    return y.astype(np.float32)


def cached_waves(sr):
    f = OUT / "cache" / f"waves_{sr}_{CFG['DURATION']}.npy"
    if f.exists():
        w = np.load(f)
        if len(w) == len(df):
            return w
    w = np.stack([load_wave(p, sr) for p in df["path"]])
    np.save(f, w)
    return w


WAVES = cached_waves(CFG["SR"])
print("Waveform cache:", WAVES.shape)

AST_FE = AutoFeatureExtractor.from_pretrained(CFG["AST_NAME"])
MEL = dict(n_fft=512, win_length=400, hop_length=160, n_mels=128)


def feat_ast(waves):
    out = [AST_FE(list(waves[i:i + 64]), sampling_rate=CFG["SR"], return_tensors="pt")["input_values"]
           for i in range(0, len(waves), 64)]
    return torch.cat(out).float()                                   # [N, 1024, 128]


def feat_logmel(waves):
    out = []
    for y in waves:
        m = librosa.feature.melspectrogram(y=y, sr=CFG["SR"], power=2.0, **MEL)
        lm = librosa.power_to_db(m, ref=np.max, top_db=80.0)          # [-80, 0] dB
        out.append(lm / 40.0 + 1.0)                                  # scaled to [-1, 1]
    return torch.tensor(np.stack(out)).unsqueeze(1).float()          # [N, 1, 128, T]


def feat_wave(waves):
    x = torch.tensor(np.asarray(waves)).float()
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)   # [N, L]


FEATURIZE = {"ast": feat_ast, "logmel": feat_logmel, "wave": feat_wave}
FEATS = {k: fn(WAVES) for k, fn in FEATURIZE.items()}

if CFG["RUN_HTSAT"]:
    from transformers import ClapFeatureExtractor, ClapAudioModel
    CLAP_FE = ClapFeatureExtractor.from_pretrained(CFG["CLAP_NAME"])

    def feat_clap(waves48):
        out = [CLAP_FE(list(waves48[i:i + 32]), sampling_rate=CFG["CLAP_SR"], truncation="rand_trunc",
                       padding="repeatpad", return_tensors="pt")["input_features"]
               for i in range(0, len(waves48), 32)]
        return torch.cat(out).float()                               # [N, 1, 1001, 64]

    FEATS["clap"] = feat_clap(cached_waves(CFG["CLAP_SR"]))

for k, v in FEATS.items():
    print(f"{k:7s} features: {tuple(v.shape)}")

# Near-duplicate detection on log-Mel statistics (cosine similarity of mean and std over time)
lm = FEATS["logmel"].squeeze(1).numpy()
emb = np.concatenate([lm.mean(2), lm.std(2)], 1)
emb = emb - emb.mean(0, keepdims=True)
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
SIM = emb @ emb.T
np.fill_diagonal(SIM, -1)
ii, jj = np.where(np.triu(SIM, 1) >= CFG["NEAR_DUP_THRESHOLD"])
NEAR_DUPS = list(zip(ii.tolist(), jj.tolist()))
pd.DataFrame([{"file_a": df.file[a], "file_b": df.file[b], "class_a": f"{df.instrument[a]}/{df.emotion[a]}",
               "class_b": f"{df.instrument[b]}/{df.emotion[b]}", "cosine": round(float(SIM[a, b]), 5)}
              for a, b in NEAR_DUPS]).to_csv(OUT / "tables" / "near_duplicates.csv", index=False)
MD5_PAIRS = [(a, b) for _, g in df.groupby("md5") if len(g) > 1
             for k, a in enumerate(g.index) for b in g.index[k + 1:]]
print(f"Near-duplicate pairs (cos >= {CFG['NEAR_DUP_THRESHOLD']}): {len(NEAR_DUPS)} | identical-file pairs: {len(MD5_PAIRS)}")

# Clips linked by a near-duplicate or identical-file pair form one group, so they never fall in different subsets
def components(pairs):
    parent = list(range(len(df)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs:
        parent[find(a)] = find(b)
    return [find(i) for i in range(len(df))]


# raise the threshold if chaining would merge too many clips into one group (largest group <= 2% of the data)
for thr in [CFG["NEAR_DUP_THRESHOLD"], 0.998, 0.999, 0.9995, 0.9999, 1.01]:
    pairs = [(a, b) for a, b in NEAR_DUPS if SIM[a, b] >= thr] + MD5_PAIRS
    root = components(pairs)
    if pd.Series(root).value_counts().max() <= max(3, int(0.02 * len(df))):
        break
print(f"Grouping threshold used: {thr}")
base = df["path"].map(make_group_id).values
first_base = {}
df["group"] = [first_base.setdefault(r, base[i]) for i, r in enumerate(root)]
sizes = df["group"].value_counts()
print(f"Groups: {df['group'].nunique()} | groups with more than one clip: {(sizes > 1).sum()} (largest {sizes.max()})")

# %% [markdown]
# ## 5. Stratified 70/10/20 split and leakage audit (Major 1, Major 4)

# %%
def make_split(seed):
    tr, va, te = CFG["SPLIT"]
    idx = np.arange(len(df))
    y = df["strata"].values
    if CFG["GROUP_SPLIT"]:
        g = df["group"].values
        s1 = StratifiedGroupKFold(n_splits=int(round(1 / te)), shuffle=True, random_state=seed)
        trv, test = next(s1.split(idx, y, g))
        s2 = StratifiedGroupKFold(n_splits=int(round((tr + va) / va)), shuffle=True, random_state=seed)
        a, b = next(s2.split(trv, y[trv], g[trv]))
        train, val = trv[a], trv[b]
    else:
        trv, test = train_test_split(idx, test_size=te, stratify=y, random_state=seed)
        train, val = train_test_split(trv, test_size=va / (tr + va), stratify=y[trv], random_state=seed)
    train, val, test = np.sort(train), np.sort(val), np.sort(test)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return train, val, test


def leakage_audit(split):
    tr, va, te = (set(s) for s in split)
    def cross(pairs, a, b):
        return sum(1 for i, j in pairs if (i in a and j in b) or (i in b and j in a))
    rep = {"identical_train_test": cross(MD5_PAIRS, tr, te), "identical_train_val": cross(MD5_PAIRS, tr, va),
           "neardup_train_test": cross(NEAR_DUPS, tr, te), "neardup_train_val": cross(NEAR_DUPS, tr, va)}
    if CFG["GROUP_SPLIT"]:
        g = df["group"].values
        rep["shared_groups_train_test"] = len(set(g[list(tr)]) & set(g[list(te)]))
    return rep


s = make_split(CFG["MAIN_SEEDS"][0])
tab = []
for name, part in zip(["Train", "Validation", "Test"], s):
    sub = df.iloc[part]
    row = {"Subset": name, "Clips": len(sub)}
    row.update({f"{i}": int((sub.instrument == i).sum()) for i in INSTRUMENTS})
    row.update({f"{e}": int((sub.emotion == e).sum()) for e in EMOTIONS})
    tab.append(row)
split_table = pd.DataFrame(tab)
split_table.to_csv(OUT / "tables" / "split_counts_seed42.csv", index=False)
print(split_table.to_string(index=False))
print("Leakage audit (seed 42):", leakage_audit(s))

# %% [markdown]
# ## 6. Data loading and SpecAugment

# %%
class CachedDS(Dataset):
    def __init__(self, idx, kind):
        self.idx = np.asarray(idx)
        self.kind = kind

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return FEATS[self.kind][j], int(Y_INST[j]), int(Y_EMO[j]), j


AXES = {"ast": (1, 2), "logmel": (3, 2), "clap": (2, 3)}   # (time axis, frequency axis) of a batch


def spec_augment(x, kind):
    if kind not in AXES:
        return x
    p = CFG["SPECAUG"]
    ta, fa = AXES[kind]
    x = x.clone()
    for b in range(x.size(0)):
        if random.random() > p["p"]:
            continue
        T, Fq = x.size(ta), x.size(fa)
        t = random.randint(0, p["time_mask"])
        if 0 < t < T:
            x[b].narrow(ta - 1, random.randint(0, T - t), t).fill_(0)
        f = random.randint(0, p["freq_mask"])
        if 0 < f < Fq:
            x[b].narrow(fa - 1, random.randint(0, Fq - f), f).fill_(0)
    return x

# %% [markdown]
# ## 7. Models: IM-ERNet and same-dataset baselines (Major 2, Minor 11, Minor 12)
#
# IM-ERNet: AST-base (12 encoder layers, 12 heads, hidden size 768, AudioSet weights) → patch tokens averaged over the
# 12 frequency rows (101 time steps) → one-layer BiLSTM with 384 units per direction (output 768) → additive attention pooling → shared block
# (LayerNorm, Dropout 0.3, Linear 768-768, GELU, Dropout 0.3) → emotion head and instrument head (4 classes each).

# %%
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, h):
        a = torch.softmax(self.w(h).squeeze(-1), dim=1)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1)


class MeanPool(nn.Module):
    def forward(self, h):
        return h.mean(1)


class MultiTaskHead(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.emo = nn.Linear(d, len(EMOTIONS))
        self.inst = nn.Linear(d, len(INSTRUMENTS))

    def forward(self, z):
        z = self.shared(z)
        return self.emo(z), self.inst(z), z


class BaseNet(nn.Module):
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last(self, n):
        raise NotImplementedError


def unfreeze_tail(module, fraction):
    params = list(module.parameters())
    for p in params[-max(1, int(len(params) * fraction)):]:
        p.requires_grad = True


class ASTNet(BaseNet):
    """IM-ERNet and its ablations. pool: 'attn', 'mean' or 'cls' (AST pooled output, used by the AST baseline)."""
    def __init__(self, use_bilstm=True, pool="attn", dropout=CFG["DROPOUT"]):
        super().__init__()
        self.backbone = ASTModel.from_pretrained(CFG["AST_NAME"])
        d = self.backbone.config.hidden_size
        self.pool_type = pool
        self.bilstm = nn.LSTM(d, d // 2, num_layers=1, batch_first=True, bidirectional=True) if use_bilstm else None
        self.pool = AttnPool(d) if pool == "attn" else (MeanPool() if pool == "mean" else None)
        self.head = MultiTaskHead(d, dropout)
        c = self.backbone.config
        self.f_dim = (c.num_mel_bins - c.patch_size) // c.frequency_stride + 1     # 12 frequency rows
        self.t_dim = (c.max_length - c.patch_size) // c.time_stride + 1            # 101 time steps

    def time_sequence(self, h):
        """Patch tokens are ordered frequency-major; regroup them on the 12 x 101 grid and average over frequency."""
        p = h[:, h.shape[1] - self.f_dim * self.t_dim:]                     # drop the two class tokens
        return p.reshape(p.shape[0], self.f_dim, self.t_dim, p.shape[-1]).mean(1)   # [B, 101, 768]

    def forward(self, x):
        out = self.backbone(input_values=x, return_dict=True)
        if self.pool_type == "cls":
            z = out.pooler_output
        else:
            h = self.time_sequence(out.last_hidden_state)   # [B, 101, 768], one vector per 100 ms step
            if self.bilstm is not None:
                h, _ = self.bilstm(h)                       # [B, 101, 768]
            z = self.pool(h)                                # [B, 768]
        return self.head(z)

    def unfreeze_last(self, n):
        bb = self.backbone
        layers = bb.encoder.layer if hasattr(bb, "encoder") else bb.layers      # transformers 4.x / 5.x
        for layer in layers[len(layers) - n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True


class CNNBiLSTMAttn(BaseNet):
    """Re-implementation of the CNN-BiLSTM-attention family [6, 13] on 128-bin log-Mel input, trained from scratch."""
    def __init__(self, dropout=CFG["DROPOUT"]):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Dropout(0.1))
        self.backbone = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128))
        self.proj = nn.Linear(128 * 16, 256)
        self.bilstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.pool = AttnPool(256)
        self.head = MultiTaskHead(256, dropout)

    def forward(self, x):
        h = self.backbone(x)                           # [B, 128, 16, T/8]
        B, C, Fq, T = h.shape
        h = F.gelu(self.proj(h.permute(0, 3, 1, 2).reshape(B, T, C * Fq)))
        h, _ = self.bilstm(h)
        return self.head(self.pool(h))


class W2VNet(BaseNet):
    """Wav2Vec2-base fine-tuned on raw audio, the backbone family of EmotionWaveNet [10]."""
    def __init__(self, dropout=CFG["DROPOUT"]):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(CFG["W2V_NAME"])
        d = self.backbone.config.hidden_size
        self.pool = AttnPool(d)
        self.head = MultiTaskHead(d, dropout)

    def forward(self, x):
        return self.head(self.pool(self.backbone(input_values=x).last_hidden_state))

    def unfreeze_last(self, n):
        layers = self.backbone.encoder.layers
        for layer in layers[len(layers) - n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.encoder.layer_norm.parameters():
            p.requires_grad = True


class HTSATNet(BaseNet):
    """HTS-AT audio encoder [21] (CLAP audio tower, AudioSet-pretrained Swin transformer)."""
    def __init__(self, dropout=CFG["DROPOUT"]):
        super().__init__()
        from transformers import ClapModel
        clap = ClapModel.from_pretrained(CFG["CLAP_NAME"])      # full CLAP checkpoint; keep the HTS-AT audio tower
        self.backbone = clap.audio_model
        del clap
        with torch.no_grad():
            d = self.backbone(input_features=FEATS["clap"][:1].to(next(self.backbone.parameters()).device)).pooler_output.shape[-1]
        self.head = MultiTaskHead(d, dropout)

    def forward(self, x):
        return self.head(self.backbone(input_features=x).pooler_output)

    def unfreeze_last(self, n):
        try:                                           # last Swin stage + final norm (about 4 of 12 blocks)
            enc = self.backbone.audio_encoder
            for p in enc.layers[-1].parameters():
                p.requires_grad = True
            for p in enc.norm.parameters():
                p.requires_grad = True
        except AttributeError:
            unfreeze_tail(self.backbone, n / 12)

# %% [markdown]
# ## 8. Experiment registry: main model, baselines and ablations (Major 2, Major 3)

# %%
def E(group, kind, build, mode="two_stage", **kw):
    d = dict(group=group, kind=kind, build=build, mode=mode, w_emo=CFG["W_EMO"], w_inst=CFG["W_INST"],
             specaug=True, unfreeze=CFG["UNFREEZE_BLOCKS"], enabled=True)
    d.update(kw)
    return d


EXPERIMENTS = {
    "IM-ERNet": E("main", "ast", lambda: ASTNet(True, "attn")),
    # baselines reproduced on the same splits, preprocessing, epochs and hardware
    "AST [3]": E("baseline", "ast", lambda: ASTNet(False, "cls")),
    "CNN-BiLSTM-Attn [6]": E("baseline", "logmel", lambda: CNNBiLSTMAttn(), mode="scratch"),
    "Wav2Vec2-base [10]": E("baseline", "wave", lambda: W2VNet(), specaug=False, enabled=CFG["RUN_W2V"]),
    "HTS-AT [21]": E("baseline", "clap", lambda: HTSATNet(), enabled=CFG["RUN_HTSAT"]),
    # ablations of IM-ERNet
    "w/o BiLSTM": E("ablation", "ast", lambda: ASTNet(False, "attn")),
    "w/o attention pooling (mean)": E("ablation", "ast", lambda: ASTNet(True, "mean")),
    "w/o multi-task (emotion only)": E("ablation", "ast", lambda: ASTNet(True, "attn"), w_inst=0.0),
    "w/o multi-task (instrument only)": E("ablation", "ast", lambda: ASTNet(True, "attn"), w_emo=0.0, enabled=False),
    "w/o Stage-1 (direct fine-tuning)": E("ablation", "ast", lambda: ASTNet(True, "attn"), mode="finetune_only"),
    "w/o Stage-2 (frozen AST)": E("ablation", "ast", lambda: ASTNet(True, "attn"), mode="frozen_only", enabled=False),
    "w/o SpecAugment": E("ablation", "ast", lambda: ASTNet(True, "attn"), specaug=False),
}
SEEDS_FOR = {"main": CFG["MAIN_SEEDS"], "baseline": CFG["BASELINE_SEEDS"], "ablation": CFG["ABLATION_SEEDS"]}
RUN_FLAG = {"main": "RUN_MAIN", "baseline": "RUN_BASELINES", "ablation": "RUN_ABLATIONS"}

# %% [markdown]
# ## 9. Training engine and multi-seed runs (Major 1, 2, 3)

# %%
def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def make_loader(idx, kind, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(CachedDS(idx, kind), batch_size=CFG["BATCH_SIZE"], shuffle=shuffle,
                      num_workers=CFG["NUM_WORKERS"], generator=g if shuffle else None)


def run_epoch(model, loader, spec, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=CFG["LABEL_SMOOTHING"] if train else 0.0)
    tot, n = 0.0, 0
    out = {"le": [], "li": [], "ye": [], "yi": [], "idx": []}
    for x, yi, ye, idx in loader:
        x, yi, ye = x.to(DEVICE, non_blocking=True), yi.to(DEVICE), ye.to(DEVICE)
        if train and spec["specaug"]:
            x = spec_augment(x, spec["kind"])
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            le, li, _ = model(x)
            loss = spec["w_emo"] * ce(le.float(), ye) + spec["w_inst"] * ce(li.float(), yi)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CFG["GRAD_CLIP"])
            scaler.step(optimizer)
            scaler.update()
        tot += loss.item() * x.size(0)
        n += x.size(0)
        for k, v in zip(["le", "li", "ye", "yi", "idx"], [le.detach().float(), li.detach().float(), ye, yi, idx]):
            out[k].append(v.cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    out["loss"] = tot / n
    out["emo_acc"] = (out["le"].argmax(1) == out["ye"]).float().mean().item()
    out["inst_acc"] = (out["li"].argmax(1) == out["yi"]).float().mean().item()
    return out


def stage_plan(mode):
    e1, e2 = CFG["EPOCHS_STAGE1"], CFG["EPOCHS_STAGE2"]
    return {"two_stage": [("frozen", e1), ("finetune", e2)], "finetune_only": [("finetune", e1 + e2)],
            "frozen_only": [("frozen", e1 + e2)], "scratch": [("scratch", e1 + e2)]}[mode]


def configure_stage(model, stage, spec):
    wd = CFG["WEIGHT_DECAY"]
    if stage == "scratch":
        for p in model.parameters():
            p.requires_grad = True
        return torch.optim.AdamW(model.parameters(), lr=CFG["LR_SCRATCH"], weight_decay=wd)
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_backbone()
    if stage == "frozen":
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["LR_STAGE1"], weight_decay=wd)
    model.unfreeze_last(spec["unfreeze"])
    bb = [p for n_, p in model.named_parameters() if p.requires_grad and n_.startswith("backbone.")]
    hd = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("backbone.")]
    return torch.optim.AdamW([{"params": bb, "lr": CFG["LR_STAGE2_BACKBONE"]},
                              {"params": hd, "lr": CFG["LR_STAGE2_HEAD"]}], weight_decay=wd)


def task_metrics(y, logits, n_classes=4):
    prob = torch.softmax(logits, 1).numpy()
    y = np.asarray(y)
    pred = prob.argmax(1)
    m = {"acc": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
         "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
         "recall_macro": recall_score(y, pred, average="macro"), "kappa": cohen_kappa_score(y, pred),
         "mcc": matthews_corrcoef(y, pred)}
    try:
        m["auc_macro"] = roc_auc_score(y, prob, multi_class="ovr", average="macro")
        m["auc_micro"] = roc_auc_score(label_binarize(y, classes=list(range(n_classes))).ravel(), prob.ravel())
    except ValueError:
        m["auc_macro"] = m["auc_micro"] = float("nan")
    return m


def fit(name, seed):
    spec = EXPERIMENTS[name]
    set_seed(seed)
    split = make_split(seed)
    tr, va, te = split
    kind = spec["kind"]
    model = spec["build"]().to(DEVICE)
    train_dl, val_dl, test_dl = make_loader(tr, kind, True, seed), make_loader(va, kind, False, seed), make_loader(te, kind, False, seed)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, hist, t0 = {"loss": float("inf")}, [], time.time()
    for stage, n_ep in stage_plan(spec["mode"]):
        opt = configure_stage(model, stage, spec)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        for ep in range(n_ep):
            t = run_epoch(model, train_dl, spec, opt, scaler)
            with torch.no_grad():
                v = run_epoch(model, val_dl, spec)
            sched.step(v["loss"])
            hist.append({"epoch": len(hist) + 1, "stage": stage, "train_loss": t["loss"], "val_loss": v["loss"],
                         "train_emo_acc": t["emo_acc"], "val_emo_acc": v["emo_acc"],
                         "train_inst_acc": t["inst_acc"], "val_inst_acc": v["inst_acc"],
                         "lr": [g["lr"] for g in opt.param_groups]})
            print(f"  [{name} | seed {seed}] ep {len(hist):02d} {stage:8s} train {t['loss']:.4f} "
                  f"val {v['loss']:.4f} | val emo {v['emo_acc']:.4f} inst {v['inst_acc']:.4f}")
            if v["loss"] < best["loss"]:
                best = {"loss": v["loss"], "epoch": len(hist), "stage": stage,
                        "state": {k: w.detach().cpu().clone() for k, w in model.state_dict().items()}}
            if stage != "frozen" or spec["mode"] == "frozen_only":
                if len(hist) - best["epoch"] >= CFG["EARLY_STOP_PATIENCE"] and len(hist) >= CFG["EPOCHS_STAGE1"] + 3:
                    print(f"  early stop at epoch {len(hist)} (best {best['epoch']})")
                    break
    train_time = time.time() - t0
    model.load_state_dict(best["state"])
    with torch.no_grad():
        v = run_epoch(model, val_dl, spec)
        te_out = run_epoch(model, test_dl, spec)

    rec = {"model": name, "group": spec["group"], "seed": seed, "kind": kind, "mode": spec["mode"],
           "n_train": len(tr), "n_val": len(va), "n_test": len(te), "best_epoch": best["epoch"],
           "best_stage": best["stage"], "train_time_s": round(train_time, 1)}
    rec.update(leakage_audit(split))
    for task, key_l, key_y, w in [("emo", "le", "ye", spec["w_emo"]), ("inst", "li", "yi", spec["w_inst"])]:
        m = task_metrics(te_out[key_y].numpy(), te_out[key_l])
        rec.update({f"{task}_{k}": (v_ if w > 0 else float("nan")) for k, v_ in m.items()})
    if spec["w_emo"] > 0 and spec["w_inst"] > 0:
        rec["joint_acc"] = float(((te_out["le"].argmax(1) == te_out["ye"]) & (te_out["li"].argmax(1) == te_out["yi"])).float().mean())
    else:
        rec["joint_acc"] = float("nan")

    np.savez_compressed(OUT / "preds" / f"{slug(name)}_seed{seed}.npz",
                        idx=te_out["idx"].numpy(), y_emo=te_out["ye"].numpy(), y_inst=te_out["yi"].numpy(),
                        logits_emo=te_out["le"].numpy(), logits_inst=te_out["li"].numpy(),
                        val_idx=v["idx"].numpy(), val_y_emo=v["ye"].numpy(), val_y_inst=v["yi"].numpy(),
                        val_logits_emo=v["le"].numpy(), val_logits_inst=v["li"].numpy())
    pd.DataFrame(hist).to_csv(OUT / "logs" / f"history_{slug(name)}_seed{seed}.csv", index=False)
    if name in CFG["SAVE_CKPT_FOR"]:
        torch.save(best["state"], OUT / "ckpt" / f"{slug(name)}_seed{seed}.pt")
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"==> {name} seed {seed}: test emo acc {rec['emo_acc']:.4f} | inst acc {rec['inst_acc']:.4f} "
          f"| best epoch {best['epoch']} | {train_time / 60:.1f} min")
    del model, best
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def done_runs():
    if not RESULTS_FILE.exists():
        return set()
    # a run counts as finished only if its predictions were saved too
    return {(r["model"], r["seed"]) for r in map(json.loads, open(RESULTS_FILE)) if r
            and (OUT / "preds" / f"{slug(r['model'])}_seed{r['seed']}.npz").exists()}


plan = [(n, s) for n, sp in EXPERIMENTS.items() if sp["enabled"] and CFG[RUN_FLAG[sp["group"]]]
        for s in SEEDS_FOR[sp["group"]]]
print(f"Planned runs: {len(plan)} | already finished: {len(done_runs() & set(plan))}")
for name, seed in plan:
    if (name, seed) in done_runs():
        continue
    fit(name, seed)

# %% [markdown]
# ## 10. Mean ± SD over seeds (Major 1, 2, 3)

# %%
res = pd.DataFrame([json.loads(l) for l in open(RESULTS_FILE) if l.strip()])
res = res.drop_duplicates(["model", "seed"], keep="last")
METRICS = ["emo_acc", "emo_f1_macro", "emo_precision_macro", "emo_recall_macro", "emo_auc_macro", "emo_kappa",
           "inst_acc", "inst_f1_macro", "inst_auc_macro", "joint_acc"]
order = [n for n in EXPERIMENTS if n in set(res["model"])]


def mean_sd(x, pct=True):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n/a"
    k = 100 if pct else 1
    return f"{k * x.mean():.2f} ± {k * x.std(ddof=1) if len(x) > 1 else 0:.2f}"


def summary(models, metrics=METRICS):
    rows = []
    for m in models:
        r = res[res["model"] == m]
        row = {"Model": m, "Runs": len(r)}
        row.update({k: mean_sd(r[k], pct=not k.endswith("kappa")) for k in metrics if k in r})
        row["Train time (min)"] = f"{r['train_time_s'].mean() / 60:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


main_base = summary([m for m in order if EXPERIMENTS[m]["group"] in ("main", "baseline")])
ablation = summary(["IM-ERNet"] + [m for m in order if EXPERIMENTS[m]["group"] == "ablation"])
main_base.to_csv(OUT / "tables" / "comparison_same_dataset_mean_sd.csv", index=False)
ablation.to_csv(OUT / "tables" / "ablation_mean_sd.csv", index=False)
res.to_csv(OUT / "tables" / "all_runs.csv", index=False)
print("Same-dataset comparison (test set, %, mean ± SD over seeds)")
print(main_base.to_string(index=False))
print("\nAblation study (test set, %, mean ± SD over seeds)")
print(ablation.to_string(index=False))


def bar(models, metric, title, fname):
    mu = [100 * res.loc[res.model == m, metric].mean() for m in models]
    sd = [100 * res.loc[res.model == m, metric].std(ddof=1) for m in models]
    plt.figure(figsize=(max(6, 1.1 * len(models)), 3.8))
    plt.bar(range(len(models)), mu, yerr=sd, capsize=4, color=["#1f77b4" if m == "IM-ERNet" else "#9ecae1" for m in models])
    plt.xticks(range(len(models)), models, rotation=30, ha="right")
    lo = np.nanmin(np.array(mu) - np.nan_to_num(sd)) - 3
    plt.ylim(max(0, lo), 100)
    plt.ylabel("Test accuracy (%)")
    plt.title(title)
    savefig(fname)


bar([m for m in order if EXPERIMENTS[m]["group"] in ("main", "baseline")], "emo_acc",
    "Emotion accuracy, same dataset (mean ± SD)", "comparison_emotion_accuracy")
abl_e = ["IM-ERNet"] + [m for m in order if EXPERIMENTS[m]["group"] == "ablation" and EXPERIMENTS[m]["w_emo"] > 0]
bar(abl_e, "emo_acc", "Ablation: emotion accuracy (mean ± SD)", "ablation_emotion_accuracy")

# Convergence curves of IM-ERNet, mean ± SD over seeds
H = [pd.read_csv(OUT / "logs" / f"history_IM_ERNet_seed{s}.csv") for s in CFG["MAIN_SEEDS"]
     if (OUT / "logs" / f"history_IM_ERNet_seed{s}.csv").exists()]
if H:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, (k, lab) in zip(ax, [("loss", "Loss"), ("emo_acc", "Emotion accuracy"), ("inst_acc", "Instrument accuracy")]):
        for split_name in ["train", "val"]:
            n_ep = max(len(h) for h in H)                      # runs end at different epochs (early stopping)
            M = np.full((len(H), n_ep), np.nan)
            for r_, h in enumerate(H):
                M[r_, :len(h)] = h[f"{split_name}_{k}"].values
            ep = np.arange(1, n_ep + 1)
            mu, sd = np.nanmean(M, 0), np.nan_to_num(np.nanstd(M, 0))
            a.plot(ep, mu, label="Training" if split_name == "train" else "Validation")
            a.fill_between(ep, mu - sd, mu + sd, alpha=0.25)
        a.axvline(CFG["EPOCHS_STAGE1"] + 0.5, ls="--", c="grey", lw=0.8)
        a.set_xlabel("Epoch"); a.set_ylabel(lab); a.legend()
    savefig("imernet_convergence_mean_sd")

# %% [markdown]
# ## 11. Statistical significance (Major 1)
#
# For each model, the comparison with IM-ERNet uses the seeds both models share. Both models were trained on the same split
# for each seed. The tests are a paired t-test and a Wilcoxon signed-rank test on the per-seed accuracies (with 5 pairs the
# smallest two-sided Wilcoxon p is 0.0625), Cohen's d_z, and an exact McNemar test on the pooled per-clip test predictions.

# %%
def load_pred(name, seed):
    f = OUT / "preds" / f"{slug(name)}_seed{seed}.npz"
    return np.load(f) if f.exists() else None


def mcnemar_exact(a_correct, b_correct):
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    p = stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
    return b01, b10, p


def significance(ref="IM-ERNet", task="emo"):
    rows = []
    ref_r = res[res.model == ref].set_index("seed")
    for m in order:
        if m == ref or res.loc[res.model == m, f"{task}_acc"].isna().all():
            continue
        r = res[res.model == m].set_index("seed")
        seeds = sorted(set(ref_r.index) & set(r.index))
        if len(seeds) < 2:
            continue
        a = ref_r.loc[seeds, f"{task}_acc"].values
        b = r.loc[seeds, f"{task}_acc"].values
        d = a - b
        t = stats.ttest_rel(a, b)
        try:
            w = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
        except ValueError:
            w = float("nan")
        ac, bc = [], []
        for s in seeds:
            pa, pb = load_pred(ref, s), load_pred(m, s)
            if pa is None or pb is None:
                continue
            assert np.array_equal(pa["idx"], pb["idx"])
            ac.append(pa[f"logits_{task}"].argmax(1) == pa[f"y_{task}"])
            bc.append(pb[f"logits_{task}"].argmax(1) == pb[f"y_{task}"])
        b01, b10, p_mc = mcnemar_exact(np.concatenate(ac), np.concatenate(bc)) if ac else (0, 0, float("nan"))
        rows.append({"Comparison": f"{ref} vs {m}", "Seeds": len(seeds), "Mean diff (pp)": round(100 * d.mean(), 2),
                     "SD diff (pp)": round(100 * d.std(ddof=1), 2), "t": round(float(t.statistic), 3),
                     "p (paired t)": float(t.pvalue), "p (Wilcoxon)": w,
                     "Cohen d_z": round(d.mean() / (d.std(ddof=1) + 1e-12), 2),
                     "McNemar n01/n10": f"{b01}/{b10}", "p (McNemar)": p_mc})
    return pd.DataFrame(rows)


for task in ["emo", "inst"]:
    sig = significance(task=task)
    sig.to_csv(OUT / "tables" / f"significance_{task}.csv", index=False)
    print(f"\nSignificance tests ({'emotion' if task == 'emo' else 'instrument'} accuracy)")
    print(sig.to_string(index=False))

# 95% confidence intervals for IM-ERNet
r = res[res.model == "IM-ERNet"]
ci_rows = []
for task in ["emo", "inst"]:
    x = r[f"{task}_acc"].values
    n = len(x)
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
        ci_rows.append({"Task": task, "Mean (%)": round(100 * x.mean(), 2), "SD (%)": round(100 * x.std(ddof=1), 2),
                        "95% CI over seeds (%)": f"[{100 * (x.mean() - h):.2f}, {100 * (x.mean() + h):.2f}]"})
    rng = np.random.default_rng(0)
    boots = []
    for s in r["seed"]:
        p = load_pred("IM-ERNet", s)
        c = (p[f"logits_{task}"].argmax(1) == p[f"y_{task}"]).astype(float)
        boots.append([c[rng.integers(0, len(c), len(c))].mean() for _ in range(2000)])
    b = np.concatenate(boots)
    if ci_rows and ci_rows[-1]["Task"] == task:
        ci_rows[-1]["95% bootstrap CI, pooled (%)"] = f"[{100 * np.percentile(b, 2.5):.2f}, {100 * np.percentile(b, 97.5):.2f}]"
ci = pd.DataFrame(ci_rows)
ci.to_csv(OUT / "tables" / "imernet_confidence_intervals.csv", index=False)
print(ci.to_string(index=False))

# %% [markdown]
# ## 12. Complexity: parameters, FLOPs, latency and memory (Major 6)
#
# Theoretical cost of IM-ERNet per clip, with N = 1214 tokens, T = 101 time steps, D = 768, L = 12 layers, H = 384 BiLSTM
# units per direction: AST encoder O(L(N²D + ND²)), BiLSTM O(T·H·(D + H)), attention pooling O(T·D), heads O(D²).
# Self-attention dominates the time. The N × N attention maps (12 heads) dominate the activation memory.

# %%
from torch.utils.flop_counter import FlopCounterMode

LSTM_FLOPS = [0]


def lstm_hook(mod, inp, out):
    x = inp[0]
    B, T = x.shape[0], x.shape[1]
    dirs = 2 if mod.bidirectional else 1
    LSTM_FLOPS[0] += 2 * B * T * dirs * 4 * mod.hidden_size * (mod.input_size + mod.hidden_size + 1)


def trainable_counts(model, spec):
    out = {}
    for stage, _ in stage_plan(spec["mode"]):
        if stage not in out:
            configure_stage(model, stage, spec)
            out[stage] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return out


@torch.no_grad()
def complexity(name):
    spec = EXPERIMENTS[name]
    model = spec["build"]().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    tc = trainable_counts(model, spec)
    model.eval()
    x1 = FEATS[spec["kind"]][:1].to(DEVICE)
    x8 = FEATS[spec["kind"]][:CFG["BATCH_SIZE"]].to(DEVICE)
    hooks = [m.register_forward_hook(lstm_hook) for m in model.modules() if isinstance(m, nn.LSTM)]
    LSTM_FLOPS[0] = 0
    try:
        with FlopCounterMode(display=False) as fc:
            model(x1)
        gflops = (fc.get_total_flops() + LSTM_FLOPS[0]) / 1e9
    except Exception as e:
        print("FLOP counting failed for", name, e)
        gflops = float("nan")
    for h in hooks:
        h.remove()

    def timed(x, reps):
        for _ in range(5):
            model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps

    lat1 = timed(x1, 50)
    lat8 = timed(x8, 20)
    peak = float("nan")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            model(x8)
        peak = torch.cuda.max_memory_allocated() / 1e6
    row = {"Model": name, "Total params (M)": round(total / 1e6, 2),
           "Trainable Stage-1 (M)": round(tc.get("frozen", tc.get("scratch", 0)) / 1e6, 2),
           "Trainable Stage-2 (M)": round(tc.get("finetune", tc.get("scratch", 0)) / 1e6, 2),
           "Size fp32 (MB)": round(total * 4 / 1e6, 1), "GFLOPs / clip": round(gflops, 2),
           "Latency / clip (ms)": round(1000 * lat1, 2), "Throughput (clips/s)": round(CFG["BATCH_SIZE"] / lat8, 1),
           "Peak GPU memory, batch 8 (MB)": round(peak, 1)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row


cx_models = ["IM-ERNet", "AST [3]", "CNN-BiLSTM-Attn [6]", "w/o BiLSTM"] + \
            [m for m in ["Wav2Vec2-base [10]", "HTS-AT [21]"] if EXPERIMENTS[m]["enabled"]]
cx = pd.DataFrame([complexity(m) for m in cx_models])
cx.to_csv(OUT / "tables" / "complexity.csv", index=False)
print(f"Hardware: {ENV.get('gpu', 'CPU')} | {ENV.get('cpu_model')} | RAM {ENV.get('ram_gb')} GB")
print(cx.to_string(index=False))

# %% [markdown]
# ## 13. Failure-case analysis: acoustic profile of errors and instrument × emotion accuracy (Major 7)

# %%
def acoustic_features(y, sr=CFG["SR"]):
    y = y[: max(1, np.max(np.nonzero(y)[0]) + 1)] if np.any(y) else y
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_fn = getattr(librosa.feature, "tempo", None) or getattr(librosa.beat, "tempo")   # librosa >= 0.10 / older
    tempo = float(np.atleast_1d(tempo_fn(onset_envelope=onset, sr=sr))[0])
    rms = librosa.feature.rms(S=S, frame_length=1024)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    h, p = librosa.effects.hpss(y)
    return {"tempo_bpm": tempo,
            "onset_rate_per_s": len(librosa.onset.onset_detect(onset_envelope=onset, sr=sr)) / (len(y) / sr),
            "spectral_centroid_hz": float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
            "spectral_bandwidth_hz": float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()),
            "spectral_rolloff_hz": float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean()),
            "spectral_flatness": float(librosa.feature.spectral_flatness(S=S).mean()),
            "zcr": float(librosa.feature.zero_crossing_rate(y).mean()),
            "rms_mean": float(rms.mean()), "dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)),
            "harmonic_ratio": float(np.sum(h ** 2) / (np.sum(h ** 2) + np.sum(p ** 2) + 1e-12))}


feat_file = OUT / "cache" / "acoustic_features.csv"
if feat_file.exists():
    AF = pd.read_csv(feat_file)
else:
    AF = pd.DataFrame([acoustic_features(w) for w in WAVES])
    AF.to_csv(feat_file, index=False)
FEAT_COLS = list(AF.columns)

# Pool IM-ERNet test predictions over seeds (each clip appears in the test set of some seeds)
recs = []
for s in CFG["MAIN_SEEDS"]:
    p = load_pred("IM-ERNet", s)
    if p is None:
        continue
    pe = torch.softmax(torch.tensor(p["logits_emo"]), 1).numpy()
    for k, j in enumerate(p["idx"]):
        recs.append({"clip": int(j), "seed": s, "true_emo": int(p["y_emo"][k]), "pred_emo": int(pe[k].argmax()),
                     "conf_emo": float(pe[k].max()), "true_inst": int(p["y_inst"][k]),
                     "pred_inst": int(p["logits_inst"][k].argmax())})
P = pd.DataFrame(recs)
P["emo_wrong"] = P.true_emo != P.pred_emo
P["inst_wrong"] = P.true_inst != P.pred_inst

# Instrument x emotion test accuracy, pooled over seeds (dependence of errors on instrument / playing style)
ie = P.assign(correct=~P.emo_wrong).pivot_table(index="true_inst", columns="true_emo", values="correct", aggfunc="mean")
ie.index = [INSTRUMENTS[i] for i in ie.index]
ie.columns = [EMOTIONS[i] for i in ie.columns]
ie.to_csv(OUT / "tables" / "instrument_x_emotion_accuracy.csv")
plt.figure(figsize=(5.2, 4))
plt.imshow(100 * ie.values, cmap="Blues", vmin=max(0, 100 * np.nanmin(ie.values) - 5), vmax=100)
for a in range(ie.shape[0]):
    for b in range(ie.shape[1]):
        plt.text(b, a, f"{100 * ie.values[a, b]:.1f}", ha="center", va="center", fontsize=9)
plt.xticks(range(ie.shape[1]), ie.columns); plt.yticks(range(ie.shape[0]), ie.index)
plt.xlabel("True emotion"); plt.ylabel("Instrument"); plt.colorbar(label="Emotion accuracy (%)")
savefig("instrument_x_emotion_accuracy")

# Pooled emotion confusion matrix with correct class names
cm = confusion_matrix(P.true_emo, P.pred_emo, labels=range(4))
cmi = confusion_matrix(P.true_inst, P.pred_inst, labels=range(4))
for mat, names, title, fname in [(cm, EMOTIONS, "Emotion (pooled over seeds)", "cm_emotion_pooled"),
                                 (cmi, INSTRUMENTS, "Instrument (pooled over seeds)", "cm_instrument_pooled")]:
    plt.figure(figsize=(4.6, 4))
    plt.imshow(mat, cmap="Blues")
    for a in range(4):
        for b in range(4):
            plt.text(b, a, mat[a, b], ha="center", va="center", color="white" if mat[a, b] > mat.max() / 2 else "black")
    plt.xticks(range(4), names, rotation=30); plt.yticks(range(4), names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    savefig(fname)
pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(OUT / "tables" / "cm_emotion_pooled.csv")
pd.DataFrame(cmi, index=INSTRUMENTS, columns=INSTRUMENTS).to_csv(OUT / "tables" / "cm_instrument_pooled.csv")

# Acoustic profile of misclassified clips against correctly classified clips of the same true class
clip = P.groupby("clip").agg(n=("seed", "size"), err_rate=("emo_wrong", "mean"), true_emo=("true_emo", "first")).reset_index()
wrong_mode = P[P.emo_wrong].groupby("clip")["pred_emo"].agg(lambda x: x.mode().iloc[0])
clip["pred_mode"] = clip["clip"].map(wrong_mode).fillna(clip["true_emo"]).astype(int)   # most frequent wrong label
clip = clip.join(AF, on="clip")
clip["instrument"] = df.instrument.values[clip["clip"].values]
clip["file"] = df.file.values[clip["clip"].values]
mis = clip[clip.err_rate >= 0.5]
ok = clip[clip.err_rate == 0]
rows = []
for e in range(4):
    a, b = mis[mis.true_emo == e], ok[ok.true_emo == e]
    if len(a) == 0:
        continue
    for c in FEAT_COLS:
        pval = stats.mannwhitneyu(a[c], b[c]).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
        rows.append({"True emotion": EMOTIONS[e], "Feature": c, "Misclassified mean": round(a[c].mean(), 3),
                     "Correct mean": round(b[c].mean(), 3), "Correct SD": round(b[c].std(), 3),
                     "z of misclassified": round((a[c].mean() - b[c].mean()) / (b[c].std() + 1e-12), 2),
                     "Mann-Whitney p": pval, "n mis / n correct": f"{len(a)}/{len(b)}"})
err_prof = pd.DataFrame(rows)
err_prof.to_csv(OUT / "tables" / "error_acoustic_profile.csv", index=False)
print(err_prof.to_string(index=False))

# Is each misclassified clip closer (standardised feature space) to the centroid of its predicted class?
Z = (AF - AF.mean()) / (AF.std() + 1e-12)
cent = {e: Z.loc[ok.loc[ok.true_emo == e, "clip"].values].mean().values for e in range(4)
        if (ok.true_emo == e).any()}
closer = [np.linalg.norm(Z.loc[c].values - cent[pm]) < np.linalg.norm(Z.loc[c].values - cent[te])
          for c, pm, te in zip(mis["clip"], mis["pred_mode"], mis["true_emo"]) if pm in cent and te in cent]
print(f"Misclassified clips closer to the centroid of the predicted emotion: {np.sum(closer)}/{len(closer)}")
mis.sort_values("err_rate", ascending=False)[["file", "instrument", "true_emo", "pred_mode", "err_rate", "n"] + FEAT_COLS] \
    .assign(true_emo=lambda d: d.true_emo.map(dict(enumerate(EMOTIONS))), pred_mode=lambda d: d.pred_mode.map(dict(enumerate(EMOTIONS)))) \
    .to_csv(OUT / "tables" / "persistent_error_clips.csv", index=False)

# t-SNE of the shared IM-ERNet embedding (first seed with a checkpoint), test errors marked
from sklearn.manifold import TSNE
ck = [s for s in CFG["MAIN_SEEDS"] if (OUT / "ckpt" / f"IM_ERNet_seed{s}.pt").exists()]
if ck:
    s0 = ck[0]
    model = EXPERIMENTS["IM-ERNet"]["build"]().to(DEVICE).eval()
    model.load_state_dict(torch.load(OUT / "ckpt" / f"IM_ERNet_seed{s0}.pt", map_location=DEVICE))
    Zs = []
    with torch.no_grad():
        for i in range(0, len(df), 16):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                Zs.append(model(FEATS["ast"][i:i + 16].to(DEVICE))[2].float().cpu())
    Zs = torch.cat(Zs).numpy()
    np.save(OUT / "cache" / f"imernet_embeddings_seed{s0}.npy", Zs)
    T2 = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(Zs)
    p = load_pred("IM-ERNet", s0)
    wrong = p["idx"][p["logits_emo"].argmax(1) != p["y_emo"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, lab, names in [(ax[0], Y_EMO, EMOTIONS), (ax[1], Y_INST, INSTRUMENTS)]:
        for k, nm in enumerate(names):
            m_ = lab == k
            a.scatter(T2[m_, 0], T2[m_, 1], s=8, label=nm, alpha=0.7)
        a.scatter(T2[wrong, 0], T2[wrong, 1], s=60, facecolors="none", edgecolors="k", label="Test emotion error")
        a.set_xticks([]); a.set_yticks([]); a.legend(markerscale=1.5)
    ax[0].set_title("Coloured by emotion"); ax[1].set_title("Coloured by instrument")
    savefig("tsne_imernet_embedding")
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 14. Robustness to additive noise (Major 7)

# %%
def make_noise(n, kind, rng):
    w = rng.standard_normal(n)
    if kind == "pink":
        f = np.fft.rfft(w)
        scale = 1 / np.sqrt(np.maximum(np.arange(len(f)), 1))
        w = np.fft.irfft(f * scale, n)
    return w / (np.std(w) + 1e-12)


def add_noise(y, snr_db, kind, rng):
    active = y[np.abs(y) > 1e-4]
    p_sig = np.mean(active ** 2) if len(active) else np.mean(y ** 2)
    noise = make_noise(len(y), kind, rng) * np.sqrt(p_sig / 10 ** (snr_db / 10))
    z = y + noise
    return (z / (np.max(np.abs(z)) + 1e-8)).astype(np.float32)


@torch.no_grad()
def predict(model, x):
    le, li = [], []
    for i in range(0, len(x), 16):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            a, b, _ = model(x[i:i + 16].to(DEVICE))
        le.append(a.float().cpu()); li.append(b.float().cpu())
    return torch.cat(le).argmax(1).numpy(), torch.cat(li).argmax(1).numpy()


rob = []
for name in CFG["SAVE_CKPT_FOR"]:
    spec = EXPERIMENTS[name]
    if spec["kind"] not in FEATURIZE:
        continue
    for s in SEEDS_FOR[spec["group"]]:
        f = OUT / "ckpt" / f"{slug(name)}_seed{s}.pt"
        if not f.exists():
            continue
        model = spec["build"]().to(DEVICE).eval()
        model.load_state_dict(torch.load(f, map_location=DEVICE))
        te = make_split(s)[2]
        conds = [("clean", None)] + [(k, snr) for k in CFG["NOISE_TYPES"] for snr in CFG["NOISE_SNRS"]]
        for kind, snr in conds:
            rng = np.random.default_rng(s)
            w = WAVES[te] if snr is None else np.stack([add_noise(WAVES[j], snr, kind, rng) for j in te])
            pe, pi = predict(model, FEATURIZE[spec["kind"]](w))
            rob.append({"Model": name, "seed": s, "noise": kind, "snr_db": snr if snr is not None else np.inf,
                        "emo_acc": accuracy_score(Y_EMO[te], pe), "inst_acc": accuracy_score(Y_INST[te], pi)})
        del model
        torch.cuda.empty_cache()
rob = pd.DataFrame(rob)
if len(rob):
    rob.to_csv(OUT / "tables" / "noise_robustness_all.csv", index=False)
    rs = rob.groupby(["Model", "noise", "snr_db"]).agg(emo_mean=("emo_acc", "mean"), emo_sd=("emo_acc", "std"),
                                                       inst_mean=("inst_acc", "mean"), inst_sd=("inst_acc", "std")).reset_index()
    rs.to_csv(OUT / "tables" / "noise_robustness_mean_sd.csv", index=False)
    print(rs.round(4).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, task in zip(ax, ["emo", "inst"]):
        for (m, k), g in rs[rs.noise != "clean"].groupby(["Model", "noise"]):
            g = g.sort_values("snr_db", ascending=False)
            clean = rs[(rs.Model == m) & (rs.noise == "clean")][f"{task}_mean"].values[0]
            x = ["clean"] + [f"{int(v)}" for v in g.snr_db]
            a.errorbar(x, 100 * np.r_[clean, g[f"{task}_mean"].values], yerr=100 * np.r_[0, g[f"{task}_sd"].fillna(0).values],
                       marker="o", capsize=3, label=f"{m}, {k}")
        a.set_xlabel("SNR (dB)"); a.set_ylabel(("Emotion" if task == "emo" else "Instrument") + " accuracy (%)")
    ax[1].legend(fontsize=7)
    savefig("noise_robustness")

# %% [markdown]
# ## 15. Annotation agreement (Major 4)
#
# Kappa can only be computed from the raw labels of each annotator. Put them in a CSV with columns `file, annotator_1,
# annotator_2, ...` (emotion names) and set `ANNOTATION_CSV`. Until then, this cell writes an empty template.

# %%
def fleiss_kappa(M):
    N, k = M.shape
    n = M.sum(1)[0]
    p = M.sum(0) / (N * n)
    P_i = (np.sum(M ** 2, 1) - n) / (n * (n - 1))
    Pbar, Pe = P_i.mean(), np.sum(p ** 2)
    return (Pbar - Pe) / (1 - Pe)


if CFG["ANNOTATION_CSV"] and Path(CFG["ANNOTATION_CSV"]).exists():
    ann = pd.read_csv(CFG["ANNOTATION_CSV"])
    cols = [c for c in ann.columns if c.lower().startswith("annotator")]
    lab = ann[cols].apply(lambda s: s.map(lambda v: canon(str(v), EMOTIONS))).map(lambda v: EMOTIONS.index(v) if v else np.nan).dropna()
    M = np.stack([np.bincount(r.astype(int), minlength=4) for r in lab.values])
    out = {"annotators": len(cols), "clips": len(lab), "fleiss_kappa": round(float(fleiss_kappa(M)), 4)}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            out[f"cohen_{cols[i]}_{cols[j]}"] = round(cohen_kappa_score(lab.iloc[:, i], lab.iloc[:, j]), 4)
    json.dump(out, open(OUT / "tables" / "annotation_agreement.json", "w"), indent=2)
    print(out)
else:
    df[["file", "instrument", "emotion"]].assign(annotator_1="", annotator_2="", annotator_3="") \
        .to_csv(OUT / "tables" / "annotation_template.csv", index=False)
    print("No annotation file given. Template written to tables/annotation_template.csv")

# %% [markdown]
# ## 16. Calibration of the softmax outputs (Major 7, over-confident errors)

# %%
def ece_score(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def fit_temperature(logits, y):
    logits, y = torch.tensor(logits), torch.tensor(y)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


cal = []
for s in CFG["MAIN_SEEDS"]:
    p = load_pred("IM-ERNet", s)
    if p is None:
        continue
    for task in ["emo", "inst"]:
        T = fit_temperature(p[f"val_logits_{task}"], p[f"val_y_{task}"])
        pr0 = torch.softmax(torch.tensor(p[f"logits_{task}"]), 1).numpy()
        pr1 = torch.softmax(torch.tensor(p[f"logits_{task}"]) / T, 1).numpy()
        y = p[f"y_{task}"]
        cal.append({"seed": s, "task": task, "T": T, "ECE_before": ece_score(pr0, y), "ECE_after": ece_score(pr1, y),
                    "NLL_before": float(F.nll_loss(torch.log(torch.tensor(pr0) + 1e-12), torch.tensor(y))),
                    "NLL_after": float(F.nll_loss(torch.log(torch.tensor(pr1) + 1e-12), torch.tensor(y)))})
cal = pd.DataFrame(cal)
if len(cal):
    cal.to_csv(OUT / "tables" / "calibration_all.csv", index=False)
    print(cal.groupby("task")[["T", "ECE_before", "ECE_after", "NLL_before", "NLL_after"]].agg(["mean", "std"]).round(4))
    p = load_pred("IM-ERNet", CFG["MAIN_SEEDS"][0])
    T = cal[(cal.seed == CFG["MAIN_SEEDS"][0]) & (cal.task == "emo")]["T"].values[0]
    plt.figure(figsize=(4.4, 4.2))
    for lab_, pr in [("Before scaling", torch.softmax(torch.tensor(p["logits_emo"]), 1).numpy()),
                     (f"After scaling (T={T:.2f})", torch.softmax(torch.tensor(p["logits_emo"]) / T, 1).numpy())]:
        conf, ok_ = pr.max(1), pr.argmax(1) == p["y_emo"]
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 0:
                xs.append(conf[m].mean()); ys.append(ok_[m].mean())
        plt.plot(xs, ys, marker="o", label=lab_)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability, emotion head"); plt.legend()
    savefig("reliability_emotion")

# %% [markdown]
# ## 17. Reproducibility package (Major 8)

# %%
req = ["torch", "transformers", "librosa", "soundfile", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]
vers = {"torch": torch.__version__, "transformers": transformers.__version__, "librosa": librosa.__version__,
        "soundfile": sf.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "matplotlib": plt.matplotlib.__version__}
open(OUT / "requirements.txt", "w").write("\n".join(f"{k}=={vers[k]}" for k in req) + "\n")
json.dump({k: (v if not callable(v) else str(v)) for k, v in CFG.items()}, open(OUT / "config.json", "w"), indent=2)

readme = f"""# IM-ERNet: code and reproduction guide

## Environment
{json.dumps(ENV, indent=2)}

Python packages: see requirements.txt. Install with `pip install -r requirements.txt`.

## Data
Indian Classical Instrument Music Emotion dataset, folder layout `<Instrument>/<Emotion>/<clip>.wav|mp3`
({len(df)} clips; counts in tables/dataset_counts.csv). Set CFG["DATA_DIR"] to the dataset root.

## Protocol
- Preprocessing: resample to {CFG['SR']} Hz, mono, silence trim (top_db={CFG['TOP_DB']}), peak normalisation, {CFG['DURATION']} s per clip.
- Split: stratified by instrument x emotion, {CFG['SPLIT']} train/validation/test, a new split for every seed.
- Seeds: main {CFG['MAIN_SEEDS']}, baselines {CFG['BASELINE_SEEDS']}, ablations {CFG['ABLATION_SEEDS']}.
- Training: Stage 1 {CFG['EPOCHS_STAGE1']} epochs with AST frozen (AdamW, lr {CFG['LR_STAGE1']}); Stage 2 {CFG['EPOCHS_STAGE2']} epochs with
  the last {CFG['UNFREEZE_BLOCKS']} AST blocks unfrozen (lr {CFG['LR_STAGE2_BACKBONE']} backbone, {CFG['LR_STAGE2_HEAD']} heads); weight decay {CFG['WEIGHT_DECAY']},
  batch {CFG['BATCH_SIZE']}, loss {CFG['W_EMO']}*L_emo + {CFG['W_INST']}*L_inst, gradient clipping {CFG['GRAD_CLIP']}, mixed precision,
  ReduceLROnPlateau (factor 0.5, patience 2). The checkpoint with the lowest validation loss is evaluated on the test set.

## Run
Open the notebook, set CFG["DATA_DIR"], run all cells. Results: results.jsonl and tables/, figures at 300 dpi in figures/.
"""
open(OUT / "README.md", "w").write(readme)
pkg = OUT.parent / "imernet_v3_package"
shutil.rmtree(pkg, ignore_errors=True)
shutil.copytree(OUT, pkg, ignore=shutil.ignore_patterns("ckpt", "cache"))   # checkpoints and caches are too large to share
shutil.make_archive(str(OUT.parent / "imernet_v3_outputs"), "zip", root_dir=pkg)
print("Archive:", OUT.parent / "imernet_v3_outputs.zip")

# %% [markdown]
# ## 18. Numbers for the revised manuscript

# %%
r = res[res.model == "IM-ERNet"]
print(f"IM-ERNet over {len(r)} seeds (test set):")
for k, lab in [("emo_acc", "Emotion accuracy"), ("emo_f1_macro", "Emotion macro-F1"), ("emo_auc_macro", "Emotion macro-AUC"),
               ("inst_acc", "Instrument accuracy"), ("inst_f1_macro", "Instrument macro-F1"), ("joint_acc", "Joint accuracy")]:
    print(f"  {lab:22s}: {mean_sd(r[k])} %")
print("\nLeakage audit summary over all runs:")
print(res[[c for c in res.columns if c.startswith(("identical_", "neardup_", "shared_groups"))]].sum())
