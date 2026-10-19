import re
from string import ascii_lowercase

import torch

from src.text_encoder.beam_search import ctc_beam_search


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        tokens = []
        previous = None
        for index in inds:
            index = int(index)
            if index != previous and index != self.blank_id:
                tokens.append(index)
            previous = index
        return self.decode(tokens)

    @property
    def blank_id(self):
        return self.char2ind[self.EMPTY_TOK]

    def greedy_decode(self, log_probs) -> str:
        return self.ctc_decode(log_probs.argmax(dim=-1))

    def beam_search_decode(self, log_probs, beam_size=10) -> str:
        tokens, _ = ctc_beam_search(log_probs, self.blank_id, beam_size)[0]
        return self.decode(tokens)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
