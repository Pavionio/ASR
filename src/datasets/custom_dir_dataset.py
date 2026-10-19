from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(
        self, audio_dir=None, transcription_dir=None, *args, data_dir=None, **kwargs
    ):
        if data_dir is not None:
            data_dir = Path(data_dir)
            audio_dir = data_dir / "audio"
            if transcription_dir is None:
                transcription_dir = data_dir / "transcriptions"

        data = []
        for path in sorted(Path(audio_dir).iterdir()):
            if not path.is_file() or path.suffix.lower() not in (
                ".mp3",
                ".wav",
                ".flac",
            ):
                continue
            info = sf.info(str(path))
            text = ""
            has_text = False
            if transcription_dir is not None:
                text_path = Path(transcription_dir) / f"{path.stem}.txt"
                if text_path.is_file():
                    text = text_path.read_text(encoding="utf-8-sig").strip()
                    has_text = True
            data.append(
                {
                    "path": str(path.resolve()),
                    "text": text,
                    "audio_len": info.duration,
                    "has_text": has_text,
                }
            )
        super().__init__(data, *args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["has_text"] = self._index[index]["has_text"]
        return item
