import argparse
from pathlib import Path

from src.metrics.utils import edit_distance
from src.text_encoder import CTCTextEncoder


def calc_metrics(target_dir, pred_dir):
    target_paths = sorted(Path(target_dir).glob("*.txt"))
    if not target_paths:
        raise ValueError("No transcriptions")

    CE = WE = chars = words = 0
    for path in target_paths:
        target = CTCTextEncoder.normalize_text(
            path.read_text(encoding="utf-8-sig").strip()
        )
        pred = CTCTextEncoder.normalize_text(
            (Path(pred_dir) / path.name).read_text(encoding="utf-8-sig").strip()
        )
        CE += edit_distance(target, pred)
        WE += edit_distance(target.split(), pred.split())
        chars += len(target)
        words += len(target.split())

    return {"CER": CE / max(1, chars), "WER": WE / max(1, words)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    args = parser.parse_args()
    for name, value in calc_metrics(args.target_dir, args.pred_dir).items():
        print(f"{name}: {value:.6f} ({value:.2%})")


if __name__ == "__main__":
    main()
