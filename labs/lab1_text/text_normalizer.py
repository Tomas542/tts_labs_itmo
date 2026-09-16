"""Russian text normalizer — skeleton for lab 1.

Brings corpus text into a form usable for training a speech synthesizer.
"""

import re
import unicodedata

import num2words


def match_case(original: str, replacement: str) -> str:
    """Переносит регистр с исходного слова на текст замены.

    Примеры:
        "г-н" -> "господин"
        "Г-н" -> "Господин"
        "Г-Н" -> "ГОСПОДИН"
        "сша" -> "сэ-шэ-а"
        "США" -> "Сэ-шэ-а" (или "СЭ-ШЭ-А", если верхний регистр)
    """
    if not original:
        return replacement

    if original[0].isupper():
        return replacement.capitalize()

    return replacement.lower()


class TextNormalizer:
    """Normalizes text in Russian.


        "!.."           -> "!"
        "«цитата»"      -> '"цитата"'
        "текст * мусор" -> "текст мусор"
        "де‑факто"      -> "де-факто"      # U+2011 -> ordinary hyphen

    **Word-changing edits.** The alignment for that utterance becomes invalid and the
    row must be dropped from the training set — but the logic itself is still needed
    for lab 5, where arbitrary user input arrives with no alignment at all::

        "в 1995 г."     -> "в тысяча девятьсот девяносто пятом году"
        "прим. автора"  -> "примечание автора"

    Example:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("Расстреливать надо таких писателей!.")
        'Расстреливать надо таких писателей!'
    """

    def __init__(self) -> None:
        """Prepare the normalizer's resources.

        Put anything expensive to build here: compiled regular expressions,
        abbreviation and contraction dictionaries, a morphological analyzer.
        Building them inside :meth:`normalize` means building them 22,200 times.
        """

        # Here goes your initialization logic
        # дефисы
        self.hyphen_cleaner = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")

        # пунктуация
        self.junk_symbols = re.compile(r'[^\w\s\.,!\?:;""\'«»…\-—\u0301]')

        # удаление повторяющихся символом пунктуации
        self.multi_punct = re.compile(r"([!?:;])[\.!\?:;]+")
        self.multi_dots = re.compile(r"\.{3,}")

        # замена ёлочек в двойные кавычки
        self.quote_replacer = re.compile(r"[«»]")

        # сокращения и топ 10 аббревиатур
        self.abbreviations = {
            r"\bт\.к\b\.?": "так как",
            r"\bт\.е\b\.?": "то есть",
            r"\bи т\.д\b\.?": "и так далее",
            r"\bи т\.п\b\.?": "и тому подобное",
            r"\bприм\. автора\b": "примечание автора",
            r"\bг-н\b": "господин",
            r"\bг-жа\b": "госпожа",
            r"\bул\.\b": "улица",
            r"\bд\.\b": "дом",
            r"минюст": "минъюст",
            r"\bКПСС\b": "ка-пэ-эс-эс",
            r"\bСНО\b": "эс-эн-о",
            r"\bСША\b": "сэ-шэ-а",
            r"\bУВД\b": "у-вэ-дэ",
            r"\bКПЭ\b": "ка-пэ-э",
            r"\bЛГУ\b": "эл-гэ-у",
            r"\bМГУ\b": "эм-гэ-у",
            r"\bТГУ\b": "тэ-гэ-у",
            r"\bВМК\b": "вэ-эм-ка",
            r"\bАХЧ\b": "а-ха-че",
        }

        self.abbr_compiled = {re.compile(pattern, re.IGNORECASE): repl for pattern, repl in self.abbreviations.items()}

        # знаки и символы
        self.symbols = {
            "%": "процент",
            "°": "градус",
            "$": "доллар",
            "€": "евро",
            "₽": "рубль",
            "+": "плюс",
            "=": "равно",
            "@": "собака",
        }
        self.symbol_cleaner = re.compile(r"[%°\$€₽\+=@]")

        # цифры
        self.digits_pattern = re.compile(r"\b\d+\b")

    def _normalize_symbols_and_punct(self, text: str) -> str:
        """Punctuation"""
        text = self.hyphen_cleaner.sub("-", text)
        text = self.symbol_cleaner.sub(lambda m: f" {self.symbols[m.group(0)]} ", text)
        text = self.quote_replacer.sub('"', text)
        text = self.junk_symbols.sub("", text)
        text = self.multi_punct.sub(r"\1", text)
        text = self.multi_dots.sub("…", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        for regex, replacement in self.abbr_compiled.items():

            def replace_with_case(match: re.Match) -> str:
                matched_text = match.group(0)
                return match_case(matched_text, replacement)  # noqa: B023

            text = regex.sub(replace_with_case, text)
        return text

    def _convert_numbers(self, text: str) -> str:
        def replace_number(match):
            num_str = match.group(0)
            try:
                # Преобразование числа в текст на русском
                return num2words(int(num_str), lang="ru")
            except Exception as _:
                return num_str

        return self.digits_pattern.sub(replace_number, text)

    def normalize(self, text: str) -> str:
        if not text:
            return text

        text = unicodedata.normalize("NFC", text)
        text = self._expand_abbreviations(text)
        text = self._convert_numbers(text)
        text = self._normalize_symbols_and_punct(text)
        text = re.sub(r"\s+", " ", text).strip()

        return text


if __name__ == "__main__":
    normalizer = TextNormalizer()
    assert normalizer.normalize("!..") == "!", normalizer.normalize("!..")
    assert normalizer.normalize("«цитата»") == '"цитата"', normalizer.normalize("«цитата»")
    assert normalizer.normalize("текст * мусор") == "текст мусор", normalizer.normalize("текст * мусор")
    assert normalizer.normalize("де‑факто") == "де-факто", normalizer.normalize("де‑факто")
