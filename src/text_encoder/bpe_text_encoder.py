from pathlib import Path

import sentencepiece
import torch

from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import ROOT_PATH


class BPETextEncoder(CTCTextEncoder):
    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = ROOT_PATH / model_path
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(model_path)
        )
        self.vocab = [self.EMPTY_TOK] + [
            self.tokenizer.id_to_piece(i)
            for i in range(self.tokenizer.get_piece_size())
        ]
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {piece: i for i, piece in self.ind2char.items()}

    def encode(self, text):
        ids = self.tokenizer.encode(self.normalize_text(text), out_type=int)
        if self.tokenizer.unk_id() in ids:
            raise ValueError
        return torch.tensor([index + 1 for index in ids], dtype=torch.long).unsqueeze(0)

    def decode(self, inds):
        ids = [int(index) - 1 for index in inds if int(index) != self.blank_id]
        return self.tokenizer.decode(ids).strip()
