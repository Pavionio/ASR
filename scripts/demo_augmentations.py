import json
import random
from pathlib import Path

import comet_ml
import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torchaudio.transforms import FrequencyMasking, TimeMasking

from src.datasets.base_dataset import BaseDataset
from src.transforms.wav_augs import AddNoise, Gain, SpeedPerturbation

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


def apply_mask(transform, features):
    for _ in range(100):
        result = transform(features.clone())
        if not torch.equal(result, features):
            return result
    return result


def make_variants(audio, features):
    speed = SpeedPerturbation(factors=(0.9,), p=1.0)
    gain = Gain(min_gain_in_db=-6.0, max_gain_in_db=-5.0, p=1.0)
    noise = AddNoise(
        min_snr_in_db=15.0,
        max_snr_in_db=16.0,
        min_f_decay=0.0,
        max_f_decay=0.0,
        p=1.0,
    )
    time_mask = TimeMasking(time_mask_param=20, p=0.05)
    frequency_mask = FrequencyMasking(freq_mask_param=16)
    original_features = features(audio)
    variants = {"original": (audio, original_features)}
    for name, transform in (("gain", gain), ("noise", noise), ("speed", speed)):
        waveform = transform(audio.clone())
        variants[name] = (waveform, features(waveform))
    variants["time_masking"] = (None, apply_mask(time_mask, original_features))
    variants["frequency_masking"] = (
        None,
        apply_mask(frequency_mask, original_features),
    )
    waveform = noise(gain(speed(audio.clone())))
    masked = apply_mask(time_mask, apply_mask(frequency_mask, features(waveform)))
    variants["full_pipeline"] = (waveform, masked)
    return variants


def make_plot(variants, text, output_path):
    minimum = min(features.min().item() for _, features in variants.values())
    maximum = max(features.max().item() for _, features in variants.values())
    fig, axes = plt.subplots(len(variants), 1, figsize=(12, 16))
    for axis, (name, (_, features)) in zip(axes, variants.items()):
        axis.imshow(
            features.squeeze(0).cpu().numpy(),
            origin="lower",
            aspect="auto",
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_title(name)
    fig.suptitle(f"text: {text}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    torch.manual_seed(42)
    data_dir = ROOT / "data/datasets/librispeech"
    output_dir = ROOT / "saved/augmentation_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "dev-clean_index.json").open() as file:
        records = json.load(file)

    selected = random.sample(records, 3)

    config = OmegaConf.load(
        ROOT / "src/configs/transforms/instance_transforms/example.yaml"
    )
    pipeline = instantiate(config.inference.get_spectrogram)
    features = nn.Sequential(*list(pipeline.children())[:-1])
    dataset = BaseDataset(selected)

    config = OmegaConf.load(ROOT / "src/configs/writer/cometml.yaml")
    experiment = comet_ml.Experiment(
        project_name=config.project_name,
        workspace=config.workspace,
        auto_metric_logging=False,
        auto_param_logging=False,
    )
    experiment.set_name("show augmentations")
    experiment.log_parameters(
        {
            "seed": 42,
            "gain_db": [-6, -5],
            "snr_db": [15, 16],
            "speed": 0.9,
            "frequency_mask_param": 16,
            "time_mask_param": 20,
            "time_mask_fraction": 0.05,
            "sample_rate": 16000,
            "normalize": False,
        }
    )

    with torch.no_grad():
        for index, record in enumerate(selected):
            audio = dataset.load_audio(record["path"])
            variants = make_variants(audio, features)
            sample_dir = output_dir / Path(record["path"]).stem
            sample_dir.mkdir(exist_ok=True)

            image_path = sample_dir / "comparison.png"
            make_plot(variants, record["text"], image_path)
            experiment.log_image(str(image_path), name=f"sample_{index + 1}", step=0)
            experiment.log_text(record["text"], metadata={"utterance": sample_dir.name})

            for name, (waveform, _) in variants.items():
                if waveform is None:
                    continue
                path = sample_dir / f"{name}.wav"
                sf.write(
                    path, waveform.squeeze(0).cpu().numpy(), 16000, subtype="FLOAT"
                )
                experiment.log_audio(
                    str(path), file_name=f"sample_{index + 1}_{name}.wav", step=0
                )
            print(sample_dir)
    experiment.end()


if __name__ == "__main__":
    main()
