"""Grapheme-to-phoneme front end — skeleton for lab 3.

Run as a script to score the module on the prepared data::

    python g2p.py

Two phone error rates are reported on the split carried in the data's own `set` column: PER over
whole `test` utterances — the front end as lab 5 will run it — and PER over `test` tokens
absent from the training lexicon, which is the out-of-vocabulary model alone.
"""
import csv
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'


SIL = '<SIL>'
_LENGTH_MARK = 'ː'
_LAB_DIR = Path(__file__).resolve().parent
_REPO_DIR = _LAB_DIR.parents[1]

_VOWEL_LETTERS = set('аеёиоуыэюя')
_VOWEL_PHONES = {
    'a', 'æ', 'e', 'ɛ', 'i', 'o', 'ɵ', 'u', 'ʉ', 'ɐ', 'ə', 'ɪ', 'ʊ', 'ɨ',
}
_STRESSED_VOWELS = {'a', 'æ', 'e', 'ɛ', 'i', 'o', 'ɵ', 'u', 'ʉ'}


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _lookup_word(token: str) -> str:
    """Lowercase a token and drop punctuation. ``<SIL>`` is not passed here."""
    text = unicodedata.normalize('NFC', str(token)).lower().strip()
    return ''.join(char for char in text if char.isalpha() or char in "-'")


def _normalize_phone(phone: str) -> str:
    return unicodedata.normalize('NFC', str(phone)).replace(_LENGTH_MARK, '')


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _load_allowed_phones() -> set[str] | None:
    path = _first_existing([
        Path('common/phones_mfa.txt'),
        Path('configs/phones_mfa.txt'),
        Path('../../common/phones_mfa.txt'),
        _REPO_DIR / 'common' / 'phones_mfa.txt',
        _REPO_DIR / 'configs' / 'phones_mfa.txt',
    ])
    if path is None:
        return None
    phones = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        for phone in line.split():
            phones.add(_normalize_phone(phone))
    phones.add(SIL)
    return phones


def _load_mfa_lexicon(allowed: set[str] | None) -> dict[str, Counter]:
    path = _first_existing([
        Path('data/russian_mfa.dict'),
        _LAB_DIR / 'data' / 'russian_mfa.dict',
    ])
    lexicon: dict[str, Counter] = defaultdict(Counter)
    if path is None:
        return lexicon
    with path.open(encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            columns = line.split('\t')
            if len(columns) >= 2:
                word = _lookup_word(columns[0])
                pronunciation = columns[-1].strip()
                probability = 1.0
                if len(columns) >= 3 and _is_float(columns[1].strip()):
                    probability = float(columns[1].strip())
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                word = _lookup_word(parts[0])
                rest = parts[1:]
                probability = 1.0
                while rest and _is_float(rest[0]):
                    if probability == 1.0:
                        probability = float(rest[0])
                    rest.pop(0)
                pronunciation = ' '.join(rest)
            if not word:
                continue
            phones = [_normalize_phone(phone) for phone in pronunciation.split()]
            if not phones or 'spn' in phones:
                continue
            if allowed is not None and any(phone not in allowed for phone in phones):
                continue
            lexicon[word][' '.join(phones)] += probability
    return lexicon


def _best_pronunciation(variants: Counter) -> list[str]:
    if not variants:
        return []
    phones, _ = variants.most_common(1)[0]
    return [_normalize_phone(phone) for phone in str(phones).split() if phone]


def _keep_allowed(phones: list[str], allowed: set[str] | None) -> list[str]:
    if allowed is None:
        return phones
    return [phone for phone in phones if phone in allowed]


class _StressPredictor:
    """Guess which vowel is stressed, using only the training lexicon."""

    def __init__(self, lexicon: dict[str, Counter]) -> None:
        self.suffix_stats = {n: defaultdict(Counter) for n in range(1, 7)}
        self.stem_stats: dict[str, Counter] = defaultdict(Counter)
        self.global_stats: Counter = Counter()
        self._fit(lexicon)

    @staticmethod
    def _vowel_letters(word: str) -> list[str]:
        return [char for char in word if char in _VOWEL_LETTERS]

    @staticmethod
    def _vowel_phones(phones: list[str]) -> list[str]:
        return [phone for phone in phones if phone in _VOWEL_PHONES]

    def _find_stress(self, word: str, phones: list[str]) -> int | None:
        letters = self._vowel_letters(word)
        vowels = self._vowel_phones(phones)
        if not letters:
            return None
        if len(letters) == 1:
            return 0
        if 'ё' in letters:
            return letters.index('ё')
        if len(letters) != len(vowels):
            return None
        candidates = [
            i for i, phone in enumerate(vowels)
            if phone in _STRESSED_VOWELS and phone != 'ɨ'
        ]
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _fit(self, lexicon: dict[str, Counter]) -> None:
        for word, variants in lexicon.items():
            phones = _best_pronunciation(variants)
            stress_index = self._find_stress(word, phones)
            if stress_index is None:
                continue
            vowel_count = len(self._vowel_letters(word))
            from_end = vowel_count - 1 - stress_index
            self.global_stats[from_end] += 1
            for n in range(1, min(6, len(word)) + 1):
                self.suffix_stats[n][word[-n:]][from_end] += 1
            for cut in range(0, min(6, max(0, len(word) - 4)) + 1):
                stem = word[:-cut] if cut else word
                if len(stem) >= 4:
                    self.stem_stats[stem][stress_index] += 1

    def predict_from_end(self, word: str) -> int:
        vowels = self._vowel_letters(word)
        if not vowels:
            return 0
        if 'ё' in vowels:
            return len(vowels) - 1 - vowels.index('ё')
        for cut in range(1, min(7, max(1, len(word) - 3))):
            stats = self.stem_stats.get(word[:-cut])
            if not stats:
                continue
            stress_from_start = stats.most_common(1)[0][0]
            if stress_from_start < len(vowels):
                return len(vowels) - 1 - stress_from_start
        for n in range(min(6, len(word)), 0, -1):
            stats = self.suffix_stats[n].get(word[-n:])
            if stats and sum(stats.values()) >= 3:
                return stats.most_common(1)[0][0]
        if self.global_stats:
            return self.global_stats.most_common(1)[0][0]
        return 0


class _OOVTranscriber:
    """Letter-to-phone rules for words missing from both dictionaries."""

    _CONSONANTS = {
        'б': ('b', 'bʲ'),
        'в': ('v', 'vʲ'),
        'г': ('ɡ', 'ɟ'),
        'д': ('d̪', 'dʲ'),
        'ж': ('ʐ', 'ʐ'),
        'з': ('z̪', 'zʲ'),
        'к': ('k', 'c'),
        'л': ('ɫ', 'ʎ'),
        'м': ('m', 'mʲ'),
        'н': ('n̪', 'ɲ'),
        'п': ('p', 'pʲ'),
        'р': ('r', 'rʲ'),
        'с': ('s̪', 'sʲ'),
        'т': ('t̪', 'tʲ'),
        'ф': ('f', 'fʲ'),
        'х': ('x', 'ç'),
        'ц': ('t̪s̪', 't̪s̪'),
        'ч': ('tɕ', 'tɕ'),
        'ш': ('ʂ', 'ʂ'),
        'щ': ('ɕ', 'ɕ'),
        'й': ('j', 'j'),
    }
    _SIMPLE = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'ɡ', 'д': 'd̪', 'е': 'e', 'ё': 'o',
        'ж': 'ʐ', 'з': 'z̪', 'и': 'i', 'й': 'j', 'к': 'k', 'л': 'ɫ', 'м': 'm',
        'н': 'n̪', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's̪', 'т': 't̪', 'у': 'u',
        'ф': 'f', 'х': 'x', 'ц': 't̪s̪', 'ч': 'tɕ', 'ш': 'ʂ', 'щ': 'ɕ', 'ы': 'ɨ',
        'э': 'ɛ', 'ю': 'u', 'я': 'a',
    }
    _VOICED = set('бвгджз')
    _VOICE_TRIGGERS = set('бгджз')
    _VOICELESS = set('пфктшсхцчщ')
    _VOICE_PAIRS = {
        'б': 'п', 'п': 'б', 'в': 'ф', 'ф': 'в', 'г': 'к', 'к': 'г',
        'д': 'т', 'т': 'д', 'ж': 'ш', 'ш': 'ж', 'з': 'с', 'с': 'з',
    }
    _SOFT_VOWELS = set('еёиюя')

    def __init__(self, lexicon: dict[str, Counter], allowed: set[str] | None) -> None:
        self.allowed = allowed
        self.stress = _StressPredictor(lexicon)

    def naive(self, word: str) -> list[str]:
        result = []
        for char in _lookup_word(word):
            if char in {'ь', 'ъ'}:
                continue
            phone = self._SIMPLE.get(char)
            if phone is not None:
                result.append(phone)
        return result

    def _stress_positions(self, word: str) -> tuple[int, int]:
        vowel_positions = [i for i, char in enumerate(word) if char in _VOWEL_LETTERS]
        if not vowel_positions:
            return -1, -1
        from_end = self.stress.predict_from_end(word)
        stress_number = len(vowel_positions) - 1 - from_end
        stress_number = max(0, min(stress_number, len(vowel_positions) - 1))
        stress_position = vowel_positions[stress_number]
        before_position = vowel_positions[stress_number - 1] if stress_number > 0 else -1
        return stress_position, before_position

    def _consonant(self, word: str, i: int, final_rules: bool) -> str | None:
        char = word[i]
        n = len(word)
        if char not in self._CONSONANTS:
            return None
        if final_rules and i + 1 < n and word[i + 1] == char:
            return None
        if final_rules and char in {'т', 'д'} and i + 1 < n and word[i + 1] == 'ч':
            return None
        actual = char
        if final_rules:
            j = i + 1
            while j < n and word[j] in {'ь', 'ъ'}:
                j += 1
            next_char = word[j] if j < n else ''
            if char in self._VOICED and (j >= n or next_char in self._VOICELESS):
                actual = self._VOICE_PAIRS.get(char, char)
            elif char in self._VOICELESS and next_char in self._VOICE_TRIGGERS:
                actual = self._VOICE_PAIRS.get(char, char)
        next_char = word[i + 1] if i + 1 < n else ''
        soft = (next_char in self._SOFT_VOWELS or next_char == 'ь') and char not in {'ж', 'ш', 'ц'}
        if (final_rules and char == 'н' and i + 2 < n and word[i + 1] in {'т', 'д'}
                and word[i + 2] in (self._SOFT_VOWELS | {'ь'})):
            soft = True
        hard_phone, soft_phone = self._CONSONANTS.get(actual, self._CONSONANTS[char])
        return soft_phone if soft else hard_phone

    def _vowel(self, word: str, i: int, stress_position: int, before_position: int,
               use_stress: bool) -> list[str]:
        char = word[i]
        result = []
        if char in {'е', 'ё', 'ю', 'я'} and (
            i == 0 or word[i - 1] in _VOWEL_LETTERS or word[i - 1] in {'ь', 'ъ'}
        ):
            result.append('j')
        if not use_stress:
            simple_vowels = {
                'а': 'a', 'о': 'o', 'э': 'ɛ', 'е': 'e', 'и': 'i', 'ы': 'ɨ',
                'у': 'u', 'ё': 'o', 'я': 'a', 'ю': 'u',
            }
            result.append(simple_vowels[char])
            return result
        stressed = i == stress_position
        before_stress = i == before_position
        previous_is_soft = (
            i > 0 and word[i - 1] in self._CONSONANTS and char in self._SOFT_VOWELS
            and word[i - 1] not in {'ж', 'ш', 'ц'}
        )
        if char == 'а':
            if stressed and previous_is_soft:
                phone = 'æ'
            elif stressed:
                phone = 'a'
            elif before_stress or i == 0:
                phone = 'ɐ'
            else:
                phone = 'ə'
        elif char == 'о':
            if stressed:
                phone = 'o'
            elif before_stress or i == 0:
                phone = 'ɐ'
            else:
                phone = 'ə'
        elif char == 'э':
            phone = 'ɛ' if stressed else 'ɪ'
        elif char == 'е':
            phone = 'e' if stressed else 'ɪ'
        elif char == 'и':
            if stressed:
                phone = 'i'
            elif i > 0 and word[i - 1] in {'ж', 'ш', 'ц'}:
                phone = 'ɨ'
            else:
                phone = 'ɪ'
        elif char == 'ы':
            phone = 'ɨ'
        elif char == 'у':
            if stressed and previous_is_soft:
                phone = 'ʉ'
            elif stressed:
                phone = 'u'
            else:
                phone = 'ʊ'
        elif char == 'ё':
            phone = 'ɵ' if previous_is_soft else 'o'
        elif char == 'я':
            if stressed and i > 0 and word[i - 1] in self._CONSONANTS:
                phone = 'æ'
            elif stressed:
                phone = 'a'
            else:
                phone = 'ɪ'
        elif char == 'ю':
            if stressed and i > 0 and word[i - 1] in self._CONSONANTS:
                phone = 'ʉ'
            elif stressed:
                phone = 'u'
            else:
                phone = 'ʊ'
        else:
            return result
        result.append(phone)
        return result

    def transcribe(self, word: str, mode: str = 'final') -> list[str]:
        word = _lookup_word(word)
        if mode == 'naive':
            return self.naive(word)
        if mode not in {'soft', 'stress', 'final'}:
            raise ValueError(f'Unknown OOV mode: {mode}')
        use_stress = mode in {'stress', 'final'}
        final_rules = mode == 'final'
        stress_position, before_position = (
            self._stress_positions(word) if use_stress else (-1, -1)
        )
        result = []
        for i, char in enumerate(word):
            if char in {'ь', 'ъ'}:
                continue
            if char in self._CONSONANTS:
                phone = self._consonant(word, i, final_rules=final_rules)
                if phone is not None:
                    result.append(phone)
                continue
            if char in _VOWEL_LETTERS:
                result.extend(self._vowel(
                    word, i, stress_position, before_position, use_stress=use_stress,
                ))
        return _keep_allowed(result, self.allowed)


class G2P:
    """Turns a sentence into a phone sequence.

    Input is one sentence as the token sequence lab 2 produces: words without punctuation,
    with ``"<SIL>"`` wherever the pause predictor placed a pause::

        ["я", "вышел", "из", "дома", <SIL>, "когда", "стемнело"]

    Punctuation is gone by this point on purpose — its only role in the pipeline is to
    predict pauses, which lab 2 has already done.


    Args:
        lexicon: `word -> Counter({phones: count})` as written by `prepare_lexicon.py`.
            Pronunciations are space-separated NFC strings. Keys are lowercase and carry no
            stress mark, so look words up accordingly.
    """

    def __init__(self, lexicon: dict[str, Counter] | None = None) -> None:
        self.lexicon = lexicon or {}
        self.oov_mode = 'final'
        self.allowed_phones = _load_allowed_phones()
        self.mfa_lexicon = _load_mfa_lexicon(self.allowed_phones)
        self._oov = _OOVTranscriber(self.lexicon, self.allowed_phones)

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        """Transcribe each token of one sentence.

        Args:
            tokens: Tokens of one sentence in order, possibly including ``"<SIL>"``.

        Returns:
            One phone list per input token, so that `len(result) == len(tokens)`. A token
            may map to an empty list, but the outer length must be preserved: the
            evaluation scores in-vocabulary, dictionary and residual tokens separately and
            needs to know which output belongs to which token. ``"<SIL>"`` maps to
            ``["<SIL>"]``.

        """
        result = []
        for token in tokens:
            token = str(token)
            if token == SIL:
                result.append([SIL])
                continue
            word = _lookup_word(token)
            if word in self.lexicon:
                phones = _best_pronunciation(self.lexicon[word])
            elif word in self.mfa_lexicon:
                phones = _best_pronunciation(self.mfa_lexicon[word])
            else:
                phones = self._oov.transcribe(word, mode=self.oov_mode)
            result.append(_keep_allowed(phones, self.allowed_phones))
        return result


def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Corpus-level PER: summed edit distance over summed reference length.

    Summing before dividing weights every phone equally. Averaging per-sentence rates
    would instead weight every sentence equally and let short sentences dominate.

    Args:
        reference: Reference phone sequences.
        hypothesis: Predicted phone sequences, same length and order.

    Returns:
        Phone error rate, or `nan` when the reference is empty.
    """
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')


def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Fraction of words whose whole phone sequence is exactly right."""
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))


def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    """Read the training table and drop tokens with no reference transcription.

    Rows whose phones are empty or contain `spn` carry no reference and are removed.

    The `set` column comes with the file. It is a rule on the utterance id — test iff the
    number divides by 5

    Args:
        path: Path to the table written by `prepare_lexicon.py`.

    Returns:
        The table with `token` and `phones` as NFC strings, in sentence order.
    """
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)

    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)


def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    """Count pronunciations per token type over the given rows.
    """
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        lexicon[token][phones] += 1
    return lexicon


def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter]) -> None:
    """Print running-text PER, and PER and word accuracy on out-of-vocabulary tokens.

    Args:
        g2p: The module under test.
        tokens: One fold of the table from :func:`load_tokens`, in sentence order.
        lexicon: The training lexicon, used only to tell in-vocabulary tokens from OOV.
    """
    sentence_ref: list[list[str]] = []
    sentence_hyp: list[list[str]] = []
    oov_ref: list[list[str]] = []
    oov_hyp: list[list[str]] = []
    in_vocab_ref: list[list[str]] = []
    in_vocab_hyp: list[list[str]] = []
    residual_ref: list[list[str]] = []
    residual_hyp: list[list[str]] = []
    mfa_lexicon = getattr(g2p, 'mfa_lexicon', {})

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False)):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]

        # <SIL> is stripped before scoring on both sides. Leaving it in would grade lab 2's
        # pause placement inside lab 3's metric. It is still passed through at runtime --
        # see :meth:`G2P.transcribe`.
        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])
        for token, ref, hyp in zip(words, reference, predicted):
            if token in lexicon:
                in_vocab_ref.append(ref)
                in_vocab_hyp.append(hyp)
            else:
                oov_ref.append(ref)
                oov_hyp.append(hyp)
                word = _lookup_word(str(token))
                if word not in lexicon and word not in mfa_lexicon:
                    residual_ref.append(ref)
                    residual_hyp.append(hyp)

    n_words = len(in_vocab_ref) + len(oov_ref)
    print(f'running text  PER {phone_error_rate(sentence_ref, sentence_hyp):.4f}')
    print(f'in-vocabulary PER {phone_error_rate(in_vocab_ref, in_vocab_hyp):.4f}  '
          f'({len(in_vocab_ref)} words, {len(in_vocab_ref) / n_words:.1%})')
    print(f'OOV           PER {phone_error_rate(oov_ref, oov_hyp):.4f}  '
          f'word accuracy {word_accuracy(oov_ref, oov_hyp):.4f}  '
          f'({len(oov_ref)} words, {len(oov_ref) / n_words:.1%})')
    print(f'residual      PER {phone_error_rate(residual_ref, residual_hyp):.4f}  '
          f'word accuracy {word_accuracy(residual_ref, residual_hyp):.4f}  '
          f'({len(residual_ref)} words, {len(residual_ref) / n_words:.1%})')


def test_g2p() -> None:
    """Build the lexicon from `train` and score the module on both folds."""
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    print(f'train {len(train)} tokens / test {len(test)} tokens')

    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')

    g2p = G2P(lexicon)

    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon)

    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)


if __name__ == '__main__':
    test_g2p()
