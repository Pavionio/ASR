from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.metrics.utils import edit_distance
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        decoder,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            decoder (LMDecoder): CTC decoder used to produce text.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms or {}

        self.text_encoder = text_encoder
        self.decoder = decoder

        # define dataloaders
        self.evaluation_dataloaders = dict(dataloaders)

        # path definition

        self.save_path = Path(save_path) if save_path is not None else None

        # define metrics
        self.metrics = metrics or {"inference": []}
        self.evaluation_metrics = MetricTracker(
            "CE",
            "WE",
            "chars",
            "words",
            *[metric.name for metric in self.metrics["inference"]],
            writer=None,
        )

        if not skip_model_load:
            # init model
            self._from_pretrained(self.cfg_trainer.from_pretrained)

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            part_logs[part] = self._inference_part(part, dataloader)
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        batch.update(self.model(**batch))
        batch["log_probs"] = batch["log_probs"].cpu()
        batch["log_probs_length"] = batch["log_probs_length"].cpu()
        batch["preds"] = []

        for i, audio_path in enumerate(batch["audio_path"]):
            length = int(batch["log_probs_length"][i])
            pred = self.decoder.decode(batch["log_probs"][i, :length])
            batch["preds"].append(pred)

            if batch.get("has_text", [True] * len(batch["text"]))[i]:
                self.has_targets = True
                target = self.text_encoder.normalize_text(batch["text"][i])
                metrics.update("CE", edit_distance(target, pred))
                metrics.update("WE", edit_distance(target.split(), pred.split()))
                metrics.update("chars", len(target))
                metrics.update("words", len(target.split()))

            if self.save_path is not None:
                output_id = Path(audio_path).stem
                if output_id in self.saved_ids:
                    raise ValueError(f"Duplicate audio ID: {output_id}")
                self.saved_ids.add(output_id)
                (self.save_path / part / f"{output_id}.txt").write_text(
                    pred, encoding="utf-8"
                )

        if all(batch.get("has_text", [True])):
            for metric in self.metrics["inference"]:
                metrics.update(metric.name, metric(**batch), n=len(batch["text"]))
        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()
        self.saved_ids = set()
        self.has_targets = False

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        metrics = self.evaluation_metrics
        logs = {
            metric.name: metrics.avg(metric.name)
            for metric in self.metrics["inference"]
        }
        if self.has_targets:
            logs["CER"] = metrics.total("CE") / max(1, metrics.total("chars"))
            logs["WER"] = metrics.total("WE") / max(1, metrics.total("words"))
        return logs
