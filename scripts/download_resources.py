import gdown

from src.utils.io_utils import ROOT_PATH

FILES = {
    "saved/full_960_bpe128/model_best.pth": "1HvkHT4qrTP_Cq4ZLMeKfkkxOghYvqW_I",
    "data/tokenizers/bpe_train-clean-100_128.model": "1sq_j3vC-6kyiQpPZzgHwYr7S2VwnTwzs",
    "data/lm/3-gram.lower.arpa": "1C6PF-QAa6CaLxXBHcMAuSTT5VPyG207E",
}


def main():
    for name, file_id in FILES.items():
        path = ROOT_PATH / name
        if path.is_file():
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".part")
        result = gdown.download(id=file_id, output=str(temporary_path), quiet=False)
        if result is None:
            raise RuntimeError(f"Download failed: {name}")
        temporary_path.replace(path)


if __name__ == "__main__":
    main()
