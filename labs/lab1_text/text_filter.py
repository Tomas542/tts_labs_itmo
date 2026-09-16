"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import re

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "data/dev_sentences.csv"


class TextFilter:
    """Decides whether an utterance is usable as a training example.

    Example:
        >>> textfilter = TextFilter()
        >>> textfilter.filter("Я вышел из дома.")
        1
        >>> textfilter.filter("Александрову Г. П.")
        0
    """

    def __init__(self) -> None:
        """Prepare the classifier's resources.

        Compiled regular expressions, abbreviation and contraction dictionaries, a
        trained model — anything that should not be rebuilt for every utterance.
        """

        # digits and english letters
        digit_eng = (
            r"[a-z]|"  # eng letters
            r"\d|"  # digits
            r"[°%₽$€]|&|@|\+|="  # non-original signs
            r"\b(?:г-жи|г-н)|"
            r"\b(?:ул|д|кв|г|прим)\.|"
            r"\b(?:ммм|гмм|хм|хе|хехе|хаха|мда|ыы|угу|ыых)\b|"
            r"минюст"
        )

        prefixes = ["К", "М", "Г", "Мк"]
        values = ["Б", "М", "Г"]
        pv_combinations = r"|".join([prefix + value for prefix, value in zip(prefixes, values, strict=False)])

        zero_width = r"[\u200b\u200c\u200d\ufeff\u00ad]"

        bad_quotes = r"[„“‘’‹›]"

        abbreviations = r"\b(итд|итп|тчк|и\s+т\s*\.\s*д\.|и\s+т\s*\.\s*п\.|т\s*\.\s*д\.|т\s*\.\s*п\.|т\s*\.\s*ч\.)\b"

        bad_spacing = r"\s+[,\.?!…]|(?:(?<!\.)\.(?!\.)|(?<!\?)!|,)\S"

        invalid_symbols = r"[<>\[\]\{\}\(\)]"

        invalid_punctuation_combos = (
            r"(?<!\.)\.\.(?!\.)|"  # ровно две точки (не часть ... или …)
            r"\.{4,}|"  # 4 и более точек подряд
            r"!{2,}|"  # 2 и более восклицательных знака
            r"\?{2,}|"  # 2 и более вопросительных знака
            r"(?<!\.)[.,][.,?!…](?!\.)|"  # точка или запятая перед другими знаками
            r"\?[.,]|"  # вопросительный знак с точкой или запятой (?. или ?,)
            r"(?<!\?)[?!][.,]"  # восклицательный перед точкой/запятой, но разрешаем ?! перед точкой (?!.)
        )
        initials = r"\b[А-ЯЁ]\.\s*(?:[А-ЯЁ]\.)?"
        emoticons = r"[:;=]-?[pdзр3()\-\[\]]"

        self.reg = re.compile(
            digit_eng
            + r"|"
            + pv_combinations
            + r"|"
            + zero_width
            + r"|"
            + bad_quotes
            + r"|"
            + abbreviations
            + r"|"
            + bad_spacing
            + r"|"
            + invalid_symbols
            + r"|"
            + invalid_punctuation_combos
            + r"|"
            + initials
            + r"|"
            + emoticons,
            re.IGNORECASE,
        )

    def filter(self, text: str) -> int:
        """Classify a single utterance.

        Args:
            text: Utterance text, already passed through :class:`TextNormalizer`.

        Returns:
            ``1`` if the text is normalized and the utterance can be used for
            training;
            ``0`` if it contains something the speaker pronounced
            differently from how it is written, and the utterance should be dropped.
        """
        label = 0 if self.reg.search(text) is not None else 1
        if label == 0:
            return label
        # заглавная
        return 0 if re.search(r"\b[А-ЯЁ]{2,}\b", text) is not None else 1


def main(textfilter: TextFilter) -> None:
    dev_files = pd.read_csv(
        DEV_SET_PATH,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        header=0,
    )

    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)

    prc = precision_score(
        dev_files["is_normalized"],
        dev_files["predicted"],
        zero_division=0,
    )
    rec = recall_score(
        dev_files["is_normalized"],
        dev_files["predicted"],
        zero_division=0,
    )
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"], zero_division=0)
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")
    print(dev_files[dev_files["is_normalized"] != dev_files["predicted"]])


if __name__ == "__main__":
    textfilter = TextFilter()
    main(textfilter)
