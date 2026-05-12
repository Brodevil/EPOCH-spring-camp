from __future__ import annotations

import importlib.util
import os
import random
import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


EMOTION_CODE_TO_NAME = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

EMOTION_NAMES = list(EMOTION_CODE_TO_NAME.values())
EMOTION_NAME_TO_ID = {name: idx for idx, name in enumerate(EMOTION_NAMES)}


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 22_050
    clip_seconds: float = 3.0
    n_fft: int = 2_048
    hop_length: int = 512
    n_mels: int = 64
    fmin: float = 30.0
    fmax: float = 8_000.0
    trim_db: int = 60
    pre_emphasis: float = 0.97
    noise_scale: float = 0.003
    pitch_shift_steps: float = 1.5

    @property
    def target_length(self) -> int:
        return int(self.sample_rate * self.clip_seconds)


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    random_seed: int = 42


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "This pipeline needs `librosa`. Install it with `%pip install librosa soundfile`."
        ) from exc
    return librosa


def _require_whisper():
    try:
        import whisper
    except ImportError as exc:
        raise ImportError(
            "Transcription needs `openai-whisper`. Install it with `%pip install openai-whisper`."
        ) from exc
    return whisper


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runtime_diagnostics(data_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "has_librosa": importlib.util.find_spec("librosa") is not None,
        "has_soundfile": importlib.util.find_spec("soundfile") is not None,
        "has_whisper": importlib.util.find_spec("whisper") is not None,
        "ffmpeg_path": shutil.which("ffmpeg"),
    }

    if data_root is not None:
        root = Path(data_root)
        actor_dirs = []
        wav_files = []
        if root.exists():
            actor_dirs = [
                path for path in root.iterdir() if path.is_dir() and path.name.lower().startswith("actor_")
            ]
            wav_files = list(root.rglob("*.wav"))

        diagnostics.update(
            {
                "data_root": str(root),
                "data_root_exists": root.exists(),
                "actor_dir_count": len(actor_dirs),
                "wav_file_count": len(wav_files),
            }
        )

    return diagnostics


def assert_runtime_ready(data_root: str | os.PathLike[str]) -> dict[str, Any]:
    diagnostics = runtime_diagnostics(data_root)
    issues: list[str] = []

    if not diagnostics["data_root_exists"]:
        issues.append(f"Dataset root was not found: {diagnostics['data_root']}")
    elif diagnostics["wav_file_count"] == 0:
        issues.append(f"No `.wav` files were found under: {diagnostics['data_root']}")

    if not diagnostics["has_librosa"]:
        issues.append(
            "`librosa` is not installed. Install it with `%pip install librosa soundfile`."
        )

    if not diagnostics["has_whisper"]:
        issues.append(
            "`openai-whisper` is not installed. Install it with `%pip install openai-whisper`."
        )

    if issues:
        if not diagnostics["ffmpeg_path"]:
            issues.append(
                "`ffmpeg` is also not on PATH, so Whisper cannot fall back to its usual CLI loader."
            )

        raise EnvironmentError(
            "Notebook preflight failed:\n- " + "\n- ".join(issues)
        )

    return diagnostics


def discover_ravdess_files(root_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(root_dir)
    actor_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.lower().startswith("actor_")
    )

    if actor_dirs:
        candidates = [wav for actor_dir in actor_dirs for wav in sorted(actor_dir.glob("*.wav"))]
    else:
        candidates = sorted(root.rglob("*.wav"))

    unique_files: list[Path] = []
    seen_stems: set[str] = set()

    for wav_path in candidates:
        if wav_path.stem in seen_stems:
            continue
        seen_stems.add(wav_path.stem)
        unique_files.append(wav_path)

    return unique_files


def build_metadata(root_dir: str | os.PathLike[str]) -> pd.DataFrame:
    rows = []

    for wav_path in discover_ravdess_files(root_dir):
        parts = wav_path.stem.split("-")
        if len(parts) != 7 or parts[2] not in EMOTION_CODE_TO_NAME:
            continue

        actor_id = int(parts[6])
        emotion_name = EMOTION_CODE_TO_NAME[parts[2]]
        rows.append(
            {
                "path": str(wav_path),
                "stem": wav_path.stem,
                "modality": parts[0],
                "vocal_channel": parts[1],
                "emotion_code": parts[2],
                "emotion": emotion_name,
                "label": EMOTION_NAME_TO_ID[emotion_name],
                "intensity": int(parts[3]),
                "statement": int(parts[4]),
                "repetition": int(parts[5]),
                "actor_id": actor_id,
                "gender": "male" if actor_id % 2 else "female",
            }
        )

    df = pd.DataFrame(rows).sort_values(["actor_id", "stem"]).reset_index(drop=True)
    if df.empty:
        raise FileNotFoundError(f"No valid RAVDESS files found under {root_dir}")
    return df


def dataset_challenges(df: pd.DataFrame) -> list[str]:
    counts = df["emotion"].value_counts()
    neutral_count = int(counts.get("neutral", 0))
    majority_count = int(counts.max())
    repeated_sentences = df["statement"].nunique()

    notes = [
        f"Class imbalance exists: neutral has {neutral_count} clips while most emotions have {majority_count}.",
        f"Only {repeated_sentences} spoken sentences are repeated across the whole dataset, so transcript-only learning is weak by design.",
        "Random splits can leak speaker-specific traits from train to test, so actor-aware evaluation is worth checking after the main experiment.",
        "The dataset is small for deep learning, which makes regularization, caching, and careful validation more important.",
    ]
    return notes


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("emotion")
        .agg(
            samples=("stem", "count"),
            actors=("actor_id", "nunique"),
            female_samples=("gender", lambda s: int((s == "female").sum())),
            male_samples=("gender", lambda s: int((s == "male").sum())),
        )
        .sort_values("samples", ascending=False)
    )


def create_splits(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        stratify=df["label"],
        random_state=seed,
    )

    relative_test = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_weights_tensor(labels: list[int] | np.ndarray, device: torch.device) -> torch.Tensor:
    label_array = np.asarray(labels)
    classes = np.arange(len(EMOTION_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=label_array)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_vocabulary(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(clean_text(text).split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counter.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int = 16) -> list[int]:
    tokens = clean_text(text).split()
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_length]]
    if len(token_ids) < max_length:
        token_ids.extend([vocab["<pad>"]] * (max_length - len(token_ids)))
    return token_ids


def attach_text_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_length: int = 16,
    min_freq: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    vocab = build_vocabulary(train_df["transcript"].fillna("").tolist(), min_freq=min_freq)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["clean_transcript"] = result["transcript"].fillna("").map(clean_text)
        result["encoded_text"] = result["clean_transcript"].map(
            lambda text: encode_text(text, vocab, max_length=max_length)
        )
        return result

    return transform(train_df), transform(val_df), transform(test_df), vocab


def transcribe_dataset(
    df: pd.DataFrame,
    transcript_csv: str | os.PathLike[str],
    model_name: str = "tiny",
    language: str = "en",
    force: bool = False,
) -> pd.DataFrame:
    transcript_path = Path(transcript_csv)

    if transcript_path.exists() and not force:
        cached = pd.read_csv(transcript_path)
        if {"path", "transcript"}.issubset(cached.columns) and len(cached) >= len(df):
            merged = df.merge(cached[["path", "transcript"]], on="path", how="left")
            if merged["transcript"].notna().all():
                return merged

    whisper = _require_whisper()
    model = whisper.load_model(model_name)
    ffmpeg_available = shutil.which("ffmpeg") is not None
    librosa = None
    if not ffmpeg_available:
        try:
            librosa = _require_librosa()
        except ImportError as exc:
            raise ImportError(
                "Whisper transcription needs either the `ffmpeg` CLI on PATH or "
                "`librosa`/`soundfile` installed for the built-in fallback loader."
            ) from exc
    sample_rate = whisper.audio.SAMPLE_RATE

    records = []
    for row in df.itertuples(index=False):
        audio_input: str | np.ndarray
        if ffmpeg_available:
            audio_input = row.path
        else:
            # Whisper normally shells out to ffmpeg; preloading keeps the notebook working
            # on Windows setups where the ffmpeg CLI is not installed.
            audio_input, _ = librosa.load(row.path, sr=sample_rate, mono=True)

        result = model.transcribe(
            audio_input,
            language=language,
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=False,
        )
        records.append({"path": row.path, "transcript": result["text"].strip()})

    transcript_df = pd.DataFrame(records)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_df.to_csv(transcript_path, index=False)
    return df.merge(transcript_df, on="path", how="left")


def _pre_emphasize(signal: np.ndarray, coeff: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def load_audio_signal(path: str, config: AudioConfig) -> np.ndarray:
    librosa = _require_librosa()
    signal, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    signal, _ = librosa.effects.trim(signal, top_db=config.trim_db)
    signal = _pre_emphasize(signal, config.pre_emphasis)

    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak

    if len(signal) < config.target_length:
        signal = np.pad(signal, (0, config.target_length - len(signal)))
    else:
        signal = signal[: config.target_length]

    return signal.astype(np.float32)


def augment_signal(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    librosa = _require_librosa()
    augmented = signal.copy()

    if np.random.rand() < 0.5:
        noise = np.random.normal(0.0, config.noise_scale, size=augmented.shape)
        augmented = augmented + noise.astype(np.float32)

    if np.random.rand() < 0.5:
        steps = np.random.uniform(-config.pitch_shift_steps, config.pitch_shift_steps)
        augmented = librosa.effects.pitch_shift(
            augmented,
            sr=config.sample_rate,
            n_steps=steps,
        )

    return augmented[: config.target_length].astype(np.float32)


def signal_to_mel_spectrogram(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    librosa = _require_librosa()
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def path_to_mel_spectrogram(path: str, config: AudioConfig, augment: bool = False) -> np.ndarray:
    signal = load_audio_signal(path, config)
    if augment:
        signal = augment_signal(signal, config)
    mel = signal_to_mel_spectrogram(signal, config)
    return np.expand_dims(mel, axis=0)


def cache_audio_features(
    df: pd.DataFrame,
    cache_dir: str | os.PathLike[str],
    config: AudioConfig,
) -> None:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for row in df.itertuples(index=False):
        cache_path = cache_root / f"{row.stem}.npy"
        if cache_path.exists():
            continue
        mel = path_to_mel_spectrogram(row.path, config, augment=False)
        np.save(cache_path, mel)


class AudioEmotionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        config: AudioConfig,
        cache_dir: str | os.PathLike[str] | None = None,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]

        if self.cache_dir and not self.augment:
            feature_path = self.cache_dir / f"{row.stem}.npy"
            mel = np.load(feature_path)
        else:
            mel = path_to_mel_spectrogram(row.path, self.config, augment=self.augment)

        x = torch.tensor(mel, dtype=torch.float32)
        y = torch.tensor(int(row.label), dtype=torch.long)
        return x, y


class TextEmotionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        x = torch.tensor(row.encoded_text, dtype=torch.long)
        y = torch.tensor(int(row.label), dtype=torch.long)
        return x, y


class MultimodalEmotionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        config: AudioConfig,
        cache_dir: str | os.PathLike[str] | None = None,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]

        if self.cache_dir and not self.augment:
            feature_path = self.cache_dir / f"{row.stem}.npy"
            mel = np.load(feature_path)
        else:
            mel = path_to_mel_spectrogram(row.path, self.config, augment=self.augment)

        audio = torch.tensor(mel, dtype=torch.float32)
        text = torch.tensor(row.encoded_text, dtype=torch.long)
        label = torch.tensor(int(row.label), dtype=torch.long)
        return audio, text, label


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 8, dropout: float = 0.35) -> None:
        super().__init__()

        self.frequency_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.time_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frequency_block(x)
        x = self.time_block(x)
        return self.bottleneck(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class TextGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 8,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.encoder(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.bottleneck(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        audio_model: AudioCNN,
        text_model: TextGRU,
        num_classes: int = 8,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.text_model = text_model

        if freeze_backbones:
            for parameter in self.audio_model.parameters():
                parameter.requires_grad = False
            for parameter in self.text_model.parameters():
                parameter.requires_grad = False

        fusion_dim = 128 + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        audio_features = self.audio_model.extract_features(audio)
        text_features = self.text_model.extract_features(text)
        joint_features = torch.cat([audio_features, text_features], dim=1)
        return self.classifier(joint_features)


def _move_batch_to_device(batch: Any, device: torch.device) -> list[torch.Tensor]:
    if isinstance(batch, (list, tuple)):
        return [item.to(device) if torch.is_tensor(item) else item for item in batch]
    raise TypeError("Expected dataloader batch to be a tuple or list.")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weights: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = {"train_loss": [], "val_loss": []}

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch_items = _move_batch_to_device(batch, device)
            *inputs, labels = batch_items

            optimizer.zero_grad()
            logits = model(*inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_items = _move_batch_to_device(batch, device)
                *inputs, labels = batch_items
                logits = model(*inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    probability_batches = []
    label_batches = []

    for batch in loader:
        batch_items = _move_batch_to_device(batch, device)
        *inputs, labels = batch_items
        logits = model(*inputs)
        probabilities = torch.softmax(logits, dim=1)

        probability_batches.append(probabilities.cpu().numpy())
        label_batches.append(labels.cpu().numpy())

    return np.concatenate(probability_batches), np.concatenate(label_batches)


def late_fusion(
    audio_probabilities: np.ndarray,
    text_probabilities: np.ndarray,
    audio_weight: float = 0.7,
) -> dict[str, np.ndarray]:
    average = (audio_probabilities + text_probabilities) / 2.0
    weighted = audio_weight * audio_probabilities + (1.0 - audio_weight) * text_probabilities

    audio_confidence = audio_probabilities.max(axis=1)
    text_confidence = text_probabilities.max(axis=1)
    choose_audio = (audio_confidence >= text_confidence)[:, None]
    max_rule = np.where(choose_audio, audio_probabilities, text_probabilities)

    return {
        "average": average,
        "weighted": weighted,
        "max_rule": max_rule,
    }


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    if label_names is None:
        label_names = EMOTION_NAMES

    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
    }


def results_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).reset_index(drop=True)


def plot_training_history(history: dict[str, list[float]], title: str = "Training History") -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    matrix: np.ndarray,
    label_names: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> None:
    if label_names is None:
        label_names = EMOTION_NAMES

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def classification_report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    report = pd.DataFrame(metrics["report"]).transpose()
    keep_rows = [name for name in EMOTION_NAMES if name in report.index]
    keep_rows.extend(["accuracy", "macro avg", "weighted avg"])
    return report.loc[keep_rows]


def actor_holdout_split(df: pd.DataFrame, holdout_actor: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["actor_id"] != holdout_actor].reset_index(drop=True)
    test_df = df[df["actor_id"] == holdout_actor].reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"Actor {holdout_actor} not found in the dataframe.")
    return train_df, test_df


def warn_if_transcripts_look_constant(df: pd.DataFrame) -> None:
    unique_texts = df["transcript"].fillna("").map(clean_text).nunique()
    if unique_texts <= 3:
        warnings.warn(
            "The transcript branch has very little lexical diversity, so text-only accuracy may stay low.",
            stacklevel=2,
        )
