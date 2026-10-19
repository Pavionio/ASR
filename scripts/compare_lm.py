import logging

import comet_ml
import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scripts.compare_decoders import evaluate
from src.datasets.collate import collate_fn
from src.text_encoder.lm_decoder import LMDecoder
from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="../src/configs", config_name="compare_lm")
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

    decoders = {}
    parameters = {}
    if config.include_no_lm:
        decoders["no_LM"] = LMDecoder(text_encoder, beam_size=config.beam_size)
        parameters["no_LM"] = {"LM": False, "alpha": 0.0, "beta": 0.0}
    for alpha in config.alphas:
        for beta in config.betas:
            name = f"LM_alpha_{alpha}_beta_{beta}"
            decoders[name] = LMDecoder(
                text_encoder, config.lm_path, config.beam_size, alpha, beta
            )
            parameters[name] = {"LM": True, "alpha": alpha, "beta": beta}

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
        rows = evaluate(model, text_encoder, dataloader, [], part, decoders)
        for row in rows:
            row["beam_size"] = config.beam_size
            row.update(parameters[row["decoder"]])
        print(pd.DataFrame(rows).to_string(index=False))
        writer.set_step(0, part)
        writer.add_table("LM_comparison", pd.DataFrame(rows))
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
