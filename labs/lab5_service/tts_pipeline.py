"""The full text-to-speech pipeline — lab 5.

Assembles labs 1–4 plus a pretrained vocoder into one callable::

    pipeline = TTSPipeline(run_dir="../lab4_acoustic/runs/baseline")
    wav, timings = pipeline("Здравствуйте, ваш заказ готов.")

The front end calls the lab 1–3 modules. The back half — acoustic model and vocoder —
is provided. A long sentence is split into chunks before FastSpeech2.

Stage order::

    text
      → normalize()        lab 1
      → tokenize()         lab 2
      → insert_pauses()    lab 2, emits <SIL>
      → to_phones()        lab 3, <SIL> passes through
      → " ".join()
      → mel()              lab 4
      → vocode()           HiFi-GAN
    audio

Pause control markup is **optional** — extra task 1. :func:`strip_markup` is here for it,
but the required path never calls it.
"""

import csv
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIL = "<SIL>"
SAMPLE_RATE = 22050
VOCODER_SOURCE = "speechbrain/tts-hifigan-libritts-22050Hz"
_CHUNK_TOKENS = 24
_BREAK = frozenset(",.!?;:…")
# Same word shape as lab 2 `label_raw`: letters or digits, hyphens inside the word allowed.
WORD_RE = re.compile(
    r"[0-9A-Za-z\u0400-\u04FF\u0301]+(?:-[0-9A-Za-z\u0400-\u04FF\u0301]+)*",
)
_LAB_DIR = Path(__file__).resolve().parent
_LABS_DIR = _LAB_DIR.parent
_REPO_DIR = _LABS_DIR.parent
_LEXICON_PATH = _LABS_DIR / "lab3_g2p" / "data" / "lexicon.csv"

# Extra task 1 only. `type="normal"` forces a pause the predictor would not place,
# `type="none"` suppresses one it would.
MARKUP_RE = re.compile(r'<pause\s+type\s*=\s*"(normal|none)"\s*/?>')
FORCE, SUPPRESS = "normal", "none"


@dataclass
class Timings:
    """Wall clock per stage, and the real-time factor of the whole call.

    RTF = synthesis time / audio duration. Below 1.0 means faster than real time: at
    RTF 2.0 a ten-second phrase takes twenty seconds to produce.
    """

    stages: dict[str, float] = field(default_factory=dict)
    audio_seconds: float = 0.0

    @property
    def total(self) -> float:
        return sum(self.stages.values())

    @property
    def rtf(self) -> float:
        return self.total / self.audio_seconds if self.audio_seconds else float("inf")

    def report(self) -> str:
        lines = [
            f"  {name:14s} {t * 1000:8.1f} ms  {t / max(self.total, 1e-9):5.1%}" for name, t in self.stages.items()
        ]
        lines.append(
            f"  {'total':14s} {self.total * 1000:8.1f} ms for "
            f"{self.audio_seconds:.2f} s of audio  ->  RTF {self.rtf:.3f}"
        )
        return "\n".join(lines)


def strip_markup(text: str) -> tuple[str, dict[int, str]]:
    """Pull pause-control tags out of the text. **Extra task 1 only.**

    Call this *before* the normalizer, not after: lab 1 lists `<` and `>` among the
    characters to reject, so a normalizer handed `<pause type="none">` would throw the
    instruction away. Real engines parse SSML before text normalization for the same
    reason.

    A tag becomes a **space**, not nothing. `одно<pause type="normal">потом` would
    otherwise collapse to `однопотом` — one word instead of two — and every pause index
    after it would be off by one.

    Args:
        text: Raw input, possibly containing `<pause type="normal">` or `type="none"`.

    Returns:
        The text with tags replaced by spaces, and `{token_index: 'normal' | 'none'}`
        where `token_index` is the **0-based index of the token the tag follows**. A tag
        before the first word has nothing to attach to and is dropped.
    """
    requests: dict[int, str] = {}
    out: list[str] = []
    cursor = 0
    for match in MARKUP_RE.finditer(text):
        out.append(text[cursor : match.start()])
        n_words = len(re.findall(r"\w+", "".join(out)))
        if n_words:  # a tag before any word has no anchor
            requests[n_words - 1] = match.group(1)
        out.append(" ")
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out), requests


def _ensure_repo_on_path() -> None:
    """Let `labs.*` import when the process was started inside `labs/lab5_service`."""
    root = str(_REPO_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_lexicon(path: Path = _LEXICON_PATH) -> dict[str, Counter]:
    """Read `word|phones|count` into the lexicon mapping `G2P` expects."""
    lexicon: dict[str, Counter] = defaultdict(Counter)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            word = (row.get("word") or "").strip()
            phones = (row.get("phones") or "").strip()
            if not word or not phones:
                continue
            lexicon[word][phones] += int(row.get("count") or 1)
    return lexicon


def tokenize_text(text: str) -> list[str]:
    """Split into lab 2 `label_raw` tokens.

    Each token is a word plus the punctuation up to the next word. Punctuation
    before the first word stays on that first token. Whitespace is not kept.
    """
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []
    tokens: list[str] = []
    for index, match in enumerate(matches):
        start = 0 if index == 0 else match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        token = text[start:end].strip()
        if token:
            tokens.append(token)
    return tokens


def chunk_tokens(tokens: list[str], limit: int = _CHUNK_TOKENS) -> list[list[str]]:
    """Split a long token sequence before FastSpeech2.

    The cut prefers `<SIL>` or a token that ends with punctuation, and a pause
    stays with the words in front of it. A sequence of `limit` tokens or fewer
    is one chunk.
    """
    if not tokens:
        return []
    if len(tokens) <= limit:
        return [list(tokens)]
    groups: list[list[str]] = []
    start = 0
    count = len(tokens)
    while start < count:
        end = min(start + limit, count)
        if end < count:
            for index in range(end - 1, start, -1):
                token = tokens[index]
                if token == SIL or token[-1] in _BREAK:
                    end = index + 1
                    break
            if end < count and tokens[end] == SIL:
                end += 1
        if end <= start:
            end = min(start + limit, count)
        groups.append(tokens[start:end])
        start = end
    return groups


def lookup_key(word: str) -> str:
    """The form to look a word up by: NFC, lowercase, no combining acute."""
    word = unicodedata.normalize("NFC", word).lower()
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", word) if c != "́"))


class TTSPipeline:
    """Text in, waveform out.

    Models load once in `__init__`, never per call. That is not tidiness: loading
    FastSpeech2 and HiFi-GAN takes seconds, and doing it per request puts the RTF far above
    1 no matter how fast the model is.

    Args:
        run_dir: A lab 4 run directory holding `config.json` and `best_model_*.pth`.
        device: `'cuda'` or `'cpu'`. RTF differs by roughly an order of magnitude.
        vocoder_source: SpeechBrain HiFi-GAN to use; the default matches the audio config
            every lab has been using.
    """

    def __init__(self, run_dir: str | Path, device: str = "cuda", vocoder_source: str = VOCODER_SOURCE) -> None:
        import torch
        from speechbrain.inference.vocoders import HIFIGAN
        from TTS.config import load_config
        from TTS.tts.configs.shared_configs import CharactersConfig
        from TTS.tts.models.forward_tts import ForwardTTS
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor

        self.device = device
        run_dir = Path(run_dir)
        config_path = next(run_dir.rglob("config.json"))
        checkpoint = sorted(run_dir.rglob("best_model*.pth"))[-1]

        config = load_config(str(config_path))
        # `load_config` cannot deserialize a list phone inventory: coqpit decides the
        # `characters` field should be a str, fails, and silently substitutes None, leaving
        # the tokenizer on coqui's default Latin set with no error. Rebuild it from the raw
        # JSON — of the checkpoint, so the vocabulary stays tied to these weights rather
        # than to whatever common/phones_mfa.txt says today.
        with open(config_path, encoding="utf-8") as fh:
            config.characters = CharactersConfig(**json.load(fh)["characters"])

        self.ap = AudioProcessor.init_from_config(config)
        self.tokenizer, self.config = TTSTokenizer.init_from_config(config)
        self.model = ForwardTTS(self.config, self.ap, self.tokenizer, speaker_manager=None)
        self.model.load_state_dict(torch.load(str(checkpoint), map_location="cpu")["model"])
        self.model.eval().to(device)

        self.vocoder = HIFIGAN.from_hparams(
            source=vocoder_source,
            savedir=str(run_dir / "pretrained_vocoder"),
            run_opts={"device": device},
        )
        self.phone_set = set(self.config.characters.characters)

        _ensure_repo_on_path()
        from labs.lab1_text.text_normalizer import TextNormalizer
        from labs.lab2_pauses.pause_predictor import PausePredictor
        from labs.lab3_g2p.g2p import G2P

        self.normalizer = TextNormalizer()
        self.pauses = PausePredictor()
        self.g2p = G2P(load_lexicon())

    # ---------------------------------------------------------------- front end (yours)
    def normalize(self, text: str) -> str:
        """Normalize the input with the lab 1 normalizer."""
        return self.normalizer.normalize(text)

    def tokenize(self, text: str) -> list[str]:
        """Split into tokens carrying their trailing punctuation.

        The pause predictor was trained on `label_raw`, so a token is a word plus
        the punctuation up to the next word.
        """
        return tokenize_text(text)

    def insert_pauses(  # noqa: ARG002
        self,
        tokens: list[str],
        requests: dict[int, str] | None = None,
    ) -> list[str]:
        """Return the token sequence with `<SIL>` where a pause belongs.

        `requests` is empty on the required path. It carries the output of
        :func:`strip_markup` for extra task 1, which this pipeline does not apply.
        """
        if not tokens:
            return []
        with_silences, _durations = self.pauses.predict_durations(tokens)
        return [str(token) for token in with_silences]

    def to_phones(self, tokens: list[str]) -> list[str]:
        """Token sequence to a flat phone sequence.

        `<SIL>` maps to itself. `G2P` drops phones outside `common/phones_mfa.txt`.
        A word with no pronunciation contributes nothing.
        """
        if not tokens:
            return []
        phones: list[str] = []
        for word in self.g2p.transcribe(tokens):
            phones.extend(word)
        return phones

    # ------------------------------------------------------------- back end (provided)
    def mel(self, phones: list[str]) -> np.ndarray:
        """Phone sequence to mel spectrogram `[n_mels, T]`.

        The phones go in space-separated because that is what
        `multicharacter_phoneme_cleaners` splits on.
        """
        import torch

        unknown = [p for p in phones if p not in self.phone_set]
        if unknown:
            msg = f"phones outside the inventory: {sorted(set(unknown))[:8]}"
            raise ValueError(msg)
        with torch.no_grad():
            out = self.model.synthesize(" ".join(phones), self.config, use_griffin_lim=False)
        return out["outputs"]["model_outputs"][0].detach().cpu().numpy().T

    def vocode(self, mel: np.ndarray) -> np.ndarray:
        """Mel `[n_mels, T]` to waveform at 22 050 Hz."""
        import torch

        with torch.no_grad():
            wav = self.vocoder.decode_batch(torch.tensor(mel).to(self.device))
        return wav.squeeze().cpu().numpy()

    # ----------------------------------------------------------------------- the whole
    def __call__(self, text: str, markup: bool = False) -> tuple[np.ndarray, Timings]:
        """Synthesize one input, timing every stage.

        Empty or whitespace-only input returns an empty waveform rather than raising: a
        service that crashes on a blank request is not a service. The same happens when
        the front end yields no phones. Token sequences longer than 24 tokens are
        synthesized in chunks and concatenated.

        Args:
            text: The input.
            markup: Parse `<pause type=...>` tags before normalizing. Off by default —
                extra task 1.
        """
        timings = Timings()

        def timed(name, fn, *args):
            start = time.perf_counter()
            result = fn(*args)
            timings.stages[name] = timings.stages.get(name, 0.0) + (time.perf_counter() - start)
            return result

        text = unicodedata.normalize("NFC", str(text or ""))
        requests: dict[int, str] = {}
        if markup:
            text, requests = timed("markup", strip_markup, text)
        if not text.strip():
            return np.zeros(0, dtype=np.float32), timings

        normalized = timed("normalize", self.normalize, text)
        tokens = timed("tokenize", self.tokenize, normalized)
        with_pauses = timed("pauses", self.insert_pauses, tokens, requests)
        parts = []
        for group in chunk_tokens(with_pauses):
            phones = timed("g2p", self.to_phones, group)
            # A single `<SIL>` still reaches FastSpeech2 when the only token has no
            # pronunciation. Coqui squeezes a length-1 encoder output and the duration
            # predictor then sees the wrong axis. No spoken phone means silence.
            if not any(phone != SIL for phone in phones):
                continue
            mel = timed("acoustic", self.mel, phones)
            waveform = np.asarray(timed("vocoder", self.vocode, mel), dtype=np.float32).reshape(-1)
            if waveform.size:
                parts.append(waveform)
        if not parts:
            return np.zeros(0, dtype=np.float32), timings

        wav = parts[0] if len(parts) == 1 else np.concatenate(parts)
        timings.audio_seconds = len(wav) / SAMPLE_RATE
        return wav, timings

    def save(self, wav: np.ndarray, path: str | Path) -> None:
        """Write a waveform with the AudioProcessor, so the sample rate matches the rest."""
        self.ap.save_wav(wav, str(path), sr=SAMPLE_RATE)
