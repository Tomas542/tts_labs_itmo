"""Pause predictor — skeleton for lab 2.

Run as a script to score the predictor on the prepared data::

    python pause_predictor.py

Precision, recall and F1 are computed for `is_pause_after`, and MAE for `pause_duration`
on true positives only. The last word of every utterance is excluded.
"""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from pymorphy3 import MorphAnalyzer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
PAUSE_PREDICTOR_DATA = "data/RUSLAN_pause_metadata.csv"
MODEL_HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
MODEL_LOGISTIC_LINEAR = "logistic_linear"
MODEL_RANDOM_FOREST = "random_forest"
MODELS = (
    MODEL_HIST_GRADIENT_BOOSTING,
    MODEL_LOGISTIC_LINEAR,
    MODEL_RANDOM_FOREST,
)

_PUNCT_NAMES = (
    "none",
    "comma",
    "period",
    "quest",
    "excl",
    "q_excl",
    "excl_q",
    "dot_q_excl",
    "ellipsis",
    "dash",
    "colon",
    "semi",
    "other",
)
_PUNCT_INDEX = {name: index for index, name in enumerate(_PUNCT_NAMES)}
_POS_NAMES = (
    "NONE",
    "NOUN",
    "ADJF",
    "ADJS",
    "COMP",
    "VERB",
    "INFN",
    "PRTF",
    "PRTS",
    "GRND",
    "NUMR",
    "ADVB",
    "NPRO",
    "PRED",
    "PREP",
    "CONJ",
    "PRCL",
    "INTJ",
)
_POS_INDEX = {name: index for index, name in enumerate(_POS_NAMES)}
_CONJUNCTIONS = {
    "и",
    "а",
    "но",
    "да",
    "или",
    "либо",
    "что",
    "чтобы",
    "когда",
    "если",
    "хотя",
    "однако",
    "зато",
    "потому",
    "поэтому",
    "то",
    "ни",
    "как",
    "чем",
}
_WORD_RE = re.compile(r"[0-9A-Za-z\u0400-\u04FF\u0301]+(?:-[0-9A-Za-z\u0400-\u04FF\u0301]+)*")
_DASH_RE = re.compile(r"(^|\s)-(\s|$)")
_CATEGORICAL = [0, 1, 2, 3, 4, 5]
_MIN_PAUSE = 0.030


def _data_path() -> Path:
    """Resolve the pause table from the working directory or this file."""
    local = Path(PAUSE_PREDICTOR_DATA)
    if local.exists():
        return local
    return Path(__file__).resolve().parent / PAUSE_PREDICTOR_DATA


def punct_class(token: str) -> str:
    """Classify the pause mark carried by one ``label_raw`` token.

    Multi-character marks are only ``!?``, ``?!``, and ``.!?``. An em dash is a
    clause break. A hyphen inside a word is not a mark.
    """
    compact = token.replace(" ", "")
    if ".!?" in compact:
        return "dot_q_excl"
    if "?!" in compact:
        return "q_excl"
    if "!?" in compact:
        return "excl_q"
    if "…" in token or "..." in token:
        return "ellipsis"
    if "!" in token:
        return "excl"
    if "?" in token:
        return "quest"
    if "." in token:
        return "period"
    if "—" in token or "–" in token or _DASH_RE.search(token):
        return "dash"
    if ":" in token:
        return "colon"
    if ";" in token:
        return "semi"
    if "," in token:
        return "comma"
    return "none"


def _word(token: str) -> str:
    match = _WORD_RE.search(token)
    if match is None:
        return ""
    return match.group().lower().replace("ё", "е")


def _code(vocabulary: dict[str, int], name: str) -> int:
    return vocabulary.get(name, 0)


class PausePredictor:
    """Predicts where pauses fall in a sentence and how long they are.

    Input is one sentence as a sequence of `label_raw` tokens — words with their
    trailing punctuation, in order::

        ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    _pos_cache: dict[str, str] = {}
    _train_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def __init__(self, model: str = MODEL_HIST_GRADIENT_BOOSTING) -> None:
        """Load the pause table and fit the placement and duration models.
        """
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}. Choose one of: {', '.join(MODELS)}")
        np.random.seed(SEED)
        self.model = model
        self._morph = MorphAnalyzer()
        self._threshold = 0.5
        self._encoder: OneHotEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._clf, self._reg = self._make_estimators(model)
        self._fit_from_table(_data_path())

    @staticmethod
    def _make_estimators(model: str):
        """Return a classifier and a duration regressor for ``model``."""
        if model == MODEL_HIST_GRADIENT_BOOSTING:
            classifier = HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=300,
                max_leaf_nodes=63,
                l2_regularization=0.1,
                categorical_features=_CATEGORICAL,
                random_state=SEED,
            )
            regressor = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.08,
                max_iter=300,
                max_leaf_nodes=63,
                l2_regularization=0.1,
                categorical_features=_CATEGORICAL,
                random_state=SEED,
            )
            return classifier, regressor
        if model == MODEL_LOGISTIC_LINEAR:
            return (
                LogisticRegression(max_iter=1000, random_state=SEED),
                LinearRegression(),
            )
        return (
            RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=SEED,
            ),
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=SEED,
            ),
        )

    def _pos(self, token: str) -> str:
        word = _word(token)
        if not word:
            return "NONE"
        cached = PausePredictor._pos_cache.get(word)
        if cached is not None:
            return cached
        parsed = self._morph.parse(word)[0].tag.POS or "NONE"
        if parsed not in _POS_INDEX:
            parsed = "NONE"
        PausePredictor._pos_cache[word] = parsed
        return parsed

    def _features(self, tokens: list[str]) -> np.ndarray:
        """Build one feature row per token."""
        count = len(tokens)

        puncts = [punct_class(token) for token in tokens]  # pause mark on this token
        poses = [self._pos(token) for token in tokens]  # pymorphy3 part of speech
        words = [_word(token) for token in tokens]  # word form; a hyphen inside the word stays

        since_words, since_chars = self._distance_since(puncts, words)
        until_words, until_chars = self._distance_until(puncts, words)

        rows = np.zeros((count, 17), dtype=np.float64)
        for index, token in enumerate(tokens):
            prev_i = index - 1
            next_i = index + 1
            rows[index, 0] = _code(_PUNCT_INDEX, puncts[index])
            rows[index, 1] = _code(_PUNCT_INDEX, puncts[prev_i] if prev_i >= 0 else "none")
            rows[index, 2] = _code(_PUNCT_INDEX, puncts[next_i] if next_i < count else "none")
            rows[index, 3] = _code(_POS_INDEX, poses[index])
            rows[index, 4] = _code(_POS_INDEX, poses[prev_i] if prev_i >= 0 else "NONE")
            rows[index, 5] = _code(_POS_INDEX, poses[next_i] if next_i < count else "NONE")
            rows[index, 6] = index / max(count - 1, 1)
            rows[index, 7] = len(words[index])
            rows[index, 8] = count
            rows[index, 9] = float(words[index] in _CONJUNCTIONS or poses[index] == "CONJ")
            next_word = words[next_i] if next_i < count else ""
            next_pos = poses[next_i] if next_i < count else "NONE"
            rows[index, 10] = float(next_word in _CONJUNCTIONS or next_pos == "CONJ")
            next_token = tokens[next_i] if next_i < count else ""
            rows[index, 11] = float(bool(next_token) and next_token[:1].isupper())
            rows[index, 12] = float(puncts[index] != "none")
            rows[index, 13] = since_words[index]
            rows[index, 14] = since_chars[index]
            rows[index, 15] = until_words[index]
            rows[index, 16] = until_chars[index]
        return rows


    @staticmethod
    def _distance_since(puncts: list[str], words: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Words and characters since the previous pause mark"""
        count = len(puncts)
        since_words = np.zeros(count, dtype=np.float64)
        since_chars = np.zeros(count, dtype=np.float64)
        running_words = 0
        running_chars = 0
        for index, punct in enumerate(puncts):
            since_words[index] = running_words
            since_chars[index] = running_chars
            if punct == "none":
                running_words += 1
                running_chars += len(words[index])
            else:
                running_words = 0
                running_chars = 0
        return since_words, since_chars

    @staticmethod
    def _distance_until(puncts: list[str], words: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Words and characters until the next pause mark"""
        count = len(puncts)
        until_words = np.zeros(count, dtype=np.float64)
        until_chars = np.zeros(count, dtype=np.float64)
        running_words = 0
        running_chars = 0
        for index in range(count - 1, -1, -1):
            until_words[index] = running_words
            until_chars[index] = running_chars
            if puncts[index] == "none":
                running_words += 1
                running_chars += len(words[index])
            else:
                running_words = 0
                running_chars = 0
        return until_words, until_chars

    def _fit_from_table(self, path: Path) -> None:
        if PausePredictor._train_cache is None:
            PausePredictor._train_cache = self._load_training_arrays(path)
        all_x, all_y, all_d, is_val = PausePredictor._train_cache
        print(f"Fitting {self.model}")
        self._threshold = self._tune_threshold(all_x[~is_val], all_y[~is_val], all_x[is_val], all_y[is_val])
        print(f"Pause threshold {self._threshold:.3f} on {len(all_y)} train tokens")
        paused = all_y == 1
        paused_x = all_x[paused]
        if self.model == MODEL_LOGISTIC_LINEAR:
            paused_x = self._encode(paused_x, fit=False)
        self._reg.fit(paused_x, all_d[paused])

    def _load_training_arrays(
        self, path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        table = pd.read_csv(path, sep="|", quoting=csv.QUOTE_NONE)
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        durations: list[np.ndarray] = []
        fold: list[np.ndarray] = []
        for _, sentence in table.groupby("id", sort=False):
            sentence = sentence.reset_index(drop=True)
            if sentence["set"].iloc[0] != "train":
                continue
            usable = sentence["is_last_word"].to_numpy() == 0
            if not usable.any():
                continue
            matrix = self._features(sentence["label_raw"].astype(str).tolist())
            features.append(matrix[usable])
            labels.append(sentence.loc[usable, "is_pause_after"].to_numpy(dtype=int))
            durations.append(sentence.loc[usable, "pause_duration"].to_numpy(dtype=float))
            numbers = sentence.loc[usable, "id"].astype(str).str.split("_").str[0].astype(int)
            fold.append((numbers.to_numpy() % 5) == 1)
        return np.vstack(features), np.concatenate(labels), np.concatenate(durations), np.concatenate(fold)

    def _encode(self, matrix: np.ndarray, fit: bool) -> np.ndarray:
        """One-hot encode punctuation and POS, and scale the numeric columns.

        Used by logistic and linear regression. Category codes are not magnitudes.
        """
        categorical = matrix[:, _CATEGORICAL].astype(int)
        numeric = matrix[:, len(_CATEGORICAL) :]
        if fit:
            self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self._scaler = StandardScaler()
            categorical_rows = self._encoder.fit_transform(categorical)
            numeric_rows = self._scaler.fit_transform(numeric)
        else:
            categorical_rows = self._encoder.transform(categorical)
            numeric_rows = self._scaler.transform(numeric)
        return np.hstack([categorical_rows, numeric_rows])

    def _tune_threshold(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
    ) -> float:
        """Tuned F1 threshold"""
        if self.model == MODEL_LOGISTIC_LINEAR:
            train_x = self._encode(train_x, fit=True)
            val_x = self._encode(val_x, fit=False)
        self._clf.fit(train_x, train_y)
        probabilities = self._clf.predict_proba(val_x)[:, 1]
        best_threshold = 0.5
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        for threshold in np.linspace(0.15, 0.75, 31):
            predicted = probabilities >= threshold
            score = f1_score(val_y, predicted)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        print(f"Validation F1: {best_f1}")
        return best_threshold

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decide for each token whether a pause follows it"""
        token_list = [str(token) for token in tokens]
        if not token_list:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        matrix = self._features(token_list)
        if self.model == MODEL_LOGISTIC_LINEAR:
            matrix = self._encode(matrix, fit=False)
        is_pause = (self._clf.predict_proba(matrix)[:, 1] >= self._threshold).astype(int)
        pause_duration = np.zeros(len(token_list), dtype=float)
        if is_pause.any():
            predicted = self._reg.predict(matrix[is_pause == 1])
            pause_duration[is_pause == 1] = np.maximum(predicted, _MIN_PAUSE)
        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Insert predicted pauses into the token sequence.

        This is the form the acoustic model consumes in labs 4 and 5.

        Args:
            tokens: Tokens of one sentence.

        Returns:
            Tokens with ``"<SIL>"`` after every predicted pause, and one duration per
            output token: seconds for ``"<SIL>"``, ``-1.0`` for words (left to the
            acoustic model).
        """
        is_pause, durations = self.predict(tokens)
        return _with_silences(tokens, is_pause, durations)


class RulePausePredictor:
    """Punctuation baseline: a pause follows every token that carries a mark.

    Duration is the mean length of train pauses of that punctuation class.
    Means are estimated without the last word of each utterance and without the test fold.
    """

    def __init__(self) -> None:
        """Load the pause table and store one mean duration per punctuation class."""
        self._mean_by_punct, self._fallback = self._fit_means(_data_path())

    @staticmethod
    def _fit_means(path: Path) -> tuple[dict[str, float], float]:
        table = pd.read_csv(path, sep="|", quoting=csv.QUOTE_NONE)
        paused = table[
            (table["set"] == "train") & (table["is_last_word"] == 0) & (table["is_pause_after"] == 1)
        ]
        classes = paused["label_raw"].astype(str).map(punct_class)
        means = paused.groupby(classes, sort=False)["pause_duration"].mean()
        mapping = {str(name): float(value) for name, value in means.items()}
        fallback = float(means.mean()) if len(means) else _MIN_PAUSE
        return mapping, fallback

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Put a pause after every token whose punctuation class is not None"""
        token_list = [str(token) for token in tokens]
        if not token_list:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        classes = [punct_class(token) for token in token_list]
        is_pause = np.array([name != "none" for name in classes], dtype=int)
        pause_duration = np.zeros(len(token_list), dtype=float)
        for index, name in enumerate(classes):
            if is_pause[index]:
                pause_duration[index] = max(self._mean_by_punct.get(name, self._fallback), _MIN_PAUSE)
        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Insert <SIL> after every punctuation mark"""
        is_pause, durations = self.predict(tokens)
        return _with_silences(tokens, is_pause, durations)


def _with_silences(
    tokens: list[str] | np.ndarray,
    is_pause: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def expand_is_pause(token: str, is_pause: int) -> list[str]:
        if bool(is_pause):
            return [token, "<SIL>"]
        return [token]

    def expand_durations(pause_duration: float) -> list[float]:
        if pause_duration > 0.0:
            return [-1.0, pause_duration]
        return [-1.0]

    tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
    durations_w_pauses = np.concatenate([expand_durations(dur) for dur in durations]).astype(np.float32)
    return tokens_w_pauses, durations_w_pauses


def calc_metrics(df: pd.DataFrame) -> None:
    """Print precision, recall and F1 for pause placement, and MAE for pause duration.

    MAE counts only rows where both the reference and the prediction have a pause.
    """
    rec =recall_score(df.is_pause_after, df.is_pause_hat)
    prc = precision_score(df.is_pause_after, df.is_pause_hat)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat)

    mae = mean_absolute_error(
        df[(df.is_pause_after==1) & (df.is_pause_hat==1)].pause_duration,
        df[(df.is_pause_after==1) & (df.is_pause_hat==1)].pause_duration_hat,
    )
    print(f"PRC: {prc}, REC: {rec}, F1: {f1}; MAE: {mae};")


def test_pause_predictor() -> None:
    """Run the predictor on every sentence and print train and test metrics.

    Expects the layout written by `prepare_training_data.py`: rows grouped by utterance
    in order, each utterance ending with its `is_last_word` row.
    """
    pause_df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep="|", quoting=csv.QUOTE_NONE)
    sentences = [
        sentence["label_raw"].astype(str).tolist()
        for _, sentence in pause_df.groupby("id", sort=False)
    ]
    if sum(len(tokens) for tokens in sentences) != len(pause_df):
        raise RuntimeError("Len mismatch between sentences and pause_df")

    runs = [(model, lambda model=model: PausePredictor(model=model)) for model in MODELS]
    runs.append(("rule", RulePausePredictor))
    for name, build in runs:
        print("=" * 10)
        print(name)
        print("=" * 10)
        predictor = build()
        is_pause_after_hat: list[int] = []
        pause_duration_hat: list[float] = []
        for tokens in tqdm.tqdm(sentences):
            is_pause_hat, pause_dur_hat = predictor.predict(tokens)
            is_pause_after_hat.extend(is_pause_hat.tolist())
            pause_duration_hat.extend(pause_dur_hat.tolist())
        scored = pause_df.copy()
        scored["is_pause_hat"] = is_pause_after_hat
        scored["pause_duration_hat"] = pause_duration_hat
        print("Calculate metrics, traning set")
        calc_metrics(scored[(scored.set == "train") & (scored.is_last_word == 0)])
        print("Calculate metrics, testing set")
        calc_metrics(scored[(scored.set == "test") & (scored.is_last_word == 0)])
    
if __name__=='__main__':
    test_pause_predictor()