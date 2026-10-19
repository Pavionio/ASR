import logging
from time import perf_counter

import comet_ml
import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.collate import collate_fn
from src.metrics.utils import edit_distance
from src.utils.io_utils import ROOT_PATH


def evaluate(model, text_encoder, dataloader, beam_sizes, part, decoders=None):
    sizes = list(decoders) if decoders is not None else [0] + list(beam_sizes)
    stats = {size: {"seconds": 0.0, "CE": 0, "WE": 0} for size in sizes}
    characters = words = 0
    audio_seconds = 0.0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=part):
            spectrogram = batch["spectrogram"].cuda()
            lengths = batch["spectrogram_length"]
            output = model(spectrogram, lengths)
            log_probs = output["log_probs"].cpu()
            output_lengths = output["log_probs_length"].cpu().tolist()
            del output, spectrogram

            for i, length in enumerate(output_lengths):
                target = text_encoder.normalize_text(batch["text"][i])
                target_words = target.split()
                characters += len(target)
                words += len(target_words)
                audio_seconds += int(batch["audio_length"][i]) / batch["sample_rate"][i]
                probs = log_probs[i, :length]

                for size in sizes:
                    start = perf_counter()
                    if decoders is not None:
                        prediction = decoders[size].decode(probs)
                    elif size == 0:
                        prediction = text_encoder.greedy_decode(probs)
                    else:
                        prediction = text_encoder.beam_search_decode(probs, size)
                    stats[size]["seconds"] += perf_counter() - start
                    CE = edit_distance(target, prediction)
                    WE = edit_distance(target_words, prediction.split())
                    stats[size]["CE"] += CE
                    stats[size]["WE"] += WE

    results = []
    for size, values in stats.items():
        results.append(
            {
                "part": part,
                "decoder": size
                if isinstance(size, str)
                else "greedy"
                if size == 0
                else f"beam_{size}",
                "beam_size": size,
                "CER": values["CE"] / max(1, characters),
                "WER": values["WE"] / max(1, words),
                "decode_seconds": values["seconds"],
                "audio_seconds": audio_seconds,
                "time per 1 second to decode": values["seconds"]
                / max(1e-8, audio_seconds),
            }
        )
    return results


@hydra.main(
    version_base=None, config_path="../src/configs", config_name="compare_decoders"
)
def main(config):
    torch.manual_seed(42)
    checkpoint_path = ROOT_PATH / config.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config = checkpoint["config"]
    text_encoder = instantiate(training_config.text_encoder)
    model = instantiate(training_config.model, n_tokens=len(text_encoder)).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    epoch = checkpoint["epoch"]
    del checkpoint
    features = instantiate(training_config.transforms.instance_transforms.inference)
    print(f"Checkpoint: {checkpoint_path}; epoch: {epoch}")

    logger = logging.getLogger(__name__)
    writer = instantiate(
        config.writer,
        logger=logger,
        project_config=OmegaConf.to_container(config, resolve=True),
        _recursive_=False,
    )
    writer.add_scalar("checkpoint_epoch", epoch)

    for part in config.parts:
        dataset = instantiate(
            {"_target_": "src.datasets.LibrispeechDataset", "part": part},
            text_encoder=text_encoder,
            instance_transforms=features,
            limit=config.limit,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
        )
        rows = evaluate(model, text_encoder, dataloader, config.beam_sizes, part)
        print(pd.DataFrame(rows).to_string(index=False))
        writer.set_step(0, part)
        writer.add_table("decoder_comparison", pd.DataFrame(rows))
        for row in rows:
            for key in (
                "CER",
                "WER",
                "decode_seconds",
                "audio_seconds",
                "time per 1 second to decode",
            ):
                writer.add_scalar(f"{key}_({row['decoder']})", row[key])
        torch.cuda.empty_cache()

    writer.exp.end()


if __name__ == "__main__":
    main()
