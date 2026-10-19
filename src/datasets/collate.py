import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    audio = [item["audio"].squeeze(0) for item in dataset_items]
    spectrograms = [
        item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items
    ]
    texts_encoded = [item["text_encoded"].reshape(-1).long() for item in dataset_items]

    result = {
        "audio": pad_sequence(audio, batch_first=True, padding_value=0.0),
        "audio_length": torch.tensor(
            [waveform.shape[0] for waveform in audio], dtype=torch.long
        ),
        "spectrogram": pad_sequence(
            spectrograms, batch_first=True, padding_value=0.0
        ).transpose(1, 2),
        "spectrogram_length": torch.tensor(
            [spectrogram.shape[0] for spectrogram in spectrograms], dtype=torch.long
        ),
        "text_encoded": pad_sequence(texts_encoded, batch_first=True, padding_value=0),
        "text_encoded_length": torch.tensor(
            [text.numel() for text in texts_encoded], dtype=torch.long
        ),
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
    for key in ("original_audio", "original_spectrogram", "spectrogram_before_masking"):
        if key in dataset_items[0]:
            result[key] = [item[key] for item in dataset_items]
    if "sample_rate" in dataset_items[0]:
        result["sample_rate"] = [item["sample_rate"] for item in dataset_items]
    if "has_text" in dataset_items[0]:
        result["has_text"] = [item["has_text"] for item in dataset_items]
    return result
