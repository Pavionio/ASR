from ctc_beam_decoder import build_ctcdecoder

from src.utils.io_utils import ROOT_PATH


class LMDecoder:
    def __init__(
        self, text_encoder, model_path=None, beam_size=20, alpha=0.5, beta=1.0
    ):
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.decoder = build_ctcdecoder(
            list(text_encoder.vocab),
            kenlm_model_path=str(ROOT_PATH / model_path) if model_path else None,
            alpha=alpha,
            beta=beta,
        )

    def decode(self, log_probs):
        text = self.decoder.decode(
            log_probs.detach().float().cpu().numpy(), beam_width=self.beam_size
        )
        return self.text_encoder.normalize_text(text)
