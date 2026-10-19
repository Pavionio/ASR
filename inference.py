import hydra
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn
from src.text_encoder.lm_decoder import LMDecoder
from src.trainer import Inferencer
from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    torch.manual_seed(config.inferencer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = ROOT_PATH / config.inferencer.from_pretrained
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config = checkpoint["config"]
    text_encoder = instantiate(training_config.text_encoder)
    model = instantiate(training_config.model, n_tokens=len(text_encoder))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    print(f"Checkpoint: {checkpoint_path}; epoch: {checkpoint['epoch']}")
    del checkpoint

    features = instantiate(training_config.transforms.instance_transforms.inference)
    dataloaders = {}
    for part, dataset_config in config.datasets.items():
        dataset = instantiate(
            dataset_config, text_encoder=text_encoder, instance_transforms=features
        )
        dataloaders[part] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
        )

    decoder = LMDecoder(
        text_encoder, config.lm_path, config.beam_size, config.alpha, config.beta
    )
    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        decoder=decoder,
        save_path=ROOT_PATH / config.inferencer.save_path,
        skip_model_load=True,
    )
    inferencer.run_inference()


if __name__ == "__main__":
    main()
