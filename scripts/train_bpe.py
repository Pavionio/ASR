import argparse
import json
from pathlib import Path
from string import ascii_lowercase

import sentencepiece as spm

from src.text_encoder.ctc_text_encoder import CTCTextEncoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        choices=["train-clean-100", "train-clean-360", "train-other-500"],
        default="train-clean-100",
    )
    parser.add_argument("--vocab-size", type=int, default=128)
    args = parser.parse_args()

    index_path = ROOT / "data/datasets/librispeech" / f"{args.part}_index.json"
    with index_path.open(encoding="utf-8") as file:
        records = json.load(file)
    texts = [CTCTextEncoder.normalize_text(record["text"]) for record in records]

    output_dir = ROOT / "data/tokenizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"bpe_{args.part}_{args.vocab_size}"
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        required_chars=ascii_lowercase,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        normalization_rule_name="identity",
        num_threads=1,
        shuffle_input_sentence=False,
    )
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {"part": args.part, "vocab_size": args.vocab_size, "records": len(texts)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(prefix.with_suffix(".model"))


if __name__ == "__main__":
    main()
