"""Train FastSpeech2 on RUSLAN phone metadata — lab 4.

Baseline (required)::

    python train_fastspeech2.py --run-name baseline

Ablations (optional), each a flag on the same script so the baseline stays the control::

    python train_fastspeech2.py --run-name pitch_energy --pitch-energy
    python train_fastspeech2.py --run-name mfa_dur --durations ../../data/mfa_durations/manifest.csv
    python train_fastspeech2.py --run-name lr3e4 --lr 3e-4 --batch-size 16

Input is the phone metadata from lab 3 — `id|text|phones`, phones space-separated, every
phone drawn from `common/phones_mfa.txt`. The third column is what the model sees; the
second is carried along only so the file stays readable.

Several audio settings differ from coqui's defaults and are **not** free choices. They are
dictated by the pretrained HiFi-GAN that lab 5 will use, and were verified by comparing
coqui's mel against SpeechBrain's frame by frame:



"""
import argparse
import csv
import hashlib
import json
import time
import os
import sys
import unicodedata
from pathlib import Path

from trainer import Trainer, TrainerArgs
from trainer.generic_utils import KeepAverage
import torch.distributed as dist

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

MULTICHAR_CLASS = "TTS.tts.utils.text.characters.MulticharPhonemes"
SAMPLE_RATE = 22050
LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parent.parent

EXPERIMENTS = {
    "baseline_g2p": ("metadata_phones_train.csv", 1e-3, []),
    "baseline_gt": ("metadata_phones_gt_train.csv", 1e-3, []),
    "lr3e4_g2p": ("metadata_phones_train.csv", 3e-4, []),
    "lr5e3_g2p": ("metadata_phones_train.csv", 5e-3, []),
    "pitch_energy_gt": ("metadata_phones_gt_train.csv", 1e-3, ["--pitch-energy"]),
    "mfa_dur_gt": ("metadata_phones_gt_mfa_train.csv", 1e-3, ["--durations"]),
}
DEFAULT_CONFIGS = list(EXPERIMENTS)[:5]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=Path("../../data/metadata_phones.csv"),
                   help="phone metadata from lab 3: id|text|phones")
    p.add_argument("--wavs", type=Path, default=Path("../../data/RUSLAN/RUSLAN"),
                   help="directory of {id}.wav at 22050 Hz")
    p.add_argument("--phones", type=Path, default=Path("../../common/phones_mfa.txt"),
                   help="the phone inventory; the model vocabulary is built from it")
    p.add_argument("--output", type=Path, default=Path("runs"),
                   help="parent directory for run directories")
    p.add_argument("--run-name", default="baseline",
                   help="subdirectory under --output; use a different one per experiment")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="coqui's Fastspeech2 default, with a NoamLR schedule")
    p.add_argument("--max-audio-seconds", type=float, default=10.0,
                   help="utterances longer than this are dropped; attention is O(n^2)")

    p.add_argument("--pitch-energy", action="store_true",
                   help="ablation: enable the pitch and energy predictors. Sets the f0 "
                        "search range for a male speaker, which the coqui defaults "
                        "(1-640 Hz) get wrong badly enough to matter")
    p.add_argument("--durations", type=Path, default=None,
                   help="ablation: a wav|npy manifest of precomputed alignments from "
                        "prepare_mfa_durations.py. Turns the internal aligner off")
    p.add_argument("--continue-path", type=Path, default=None)
    p.add_argument("--total-epochs", type=int, default=None)
    p.add_argument("--num-loader-workers", type=int, default=0)
    p.add_argument("--num-eval-loader-workers", type=int, default=0)
    p.add_argument("--precompute-num-workers", type=int, default=0)
    p.add_argument("--scheduler-per-step", action="store_true")
    p.add_argument("--save-step", type=int, default=0)
    p.add_argument("--save-n-checkpoints", type=int, default=2)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--archive-first-epochs", type=int, default=0)
    p.add_argument("--archive-every-epochs", type=int, default=0)
    p.add_argument("--suite", action="store_true",
                   help="run the experiment list in this process instead of one run")
    p.add_argument("--resume", action="store_true",
                   help="with --suite, continue the existing experiment directories")
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=None,
                   help="with --suite, which experiments to run")
    p.add_argument("--include-mfa", action="store_true",
                   help="with --suite, also train the MFA duration experiment")
    p.add_argument("--dry-run", action="store_true",
                   help="with --suite, print the runs and do not train")
    return p.parse_args()


def load_phone_inventory(path: Path) -> list[str]:
    """Read one phone per line, NFC-normalized.

    NFC matters: `s̪` is two codepoints and `t̪s̪` is four. An NFD spelling here against
    NFC in the metadata compares unequal, every lookup misses, and the only symptom is a
    model that trains on unknown tokens.
    """
    phones = [unicodedata.normalize("NFC", line.strip())
              for line in path.read_text(encoding="utf-8").splitlines()]
    phones = [p for p in phones if p]
    if not phones:
        msg = f"No phones read from {path}"
        raise ValueError(msg)
    return phones


def make_formatter(wav_dir: Path):
    """Build a coqui sample formatter over `id|text|phones`.

    Column 3 is the model input. Rows whose wav is missing are skipped and counted rather
    than raising, because a partially downloaded corpus is the common case.
    """

    def formatter(root_path, meta_file, **kwargs):  # noqa: ARG001 - coqui's signature
        items = []
        skipped = 0
        with Path(meta_file).open(encoding="utf-8") as fh:
            for line in fh:
                cols = line.rstrip("\n").split("|")
                if len(cols) < 3:
                    skipped += 1
                    continue
                utt_id, _text, phones = cols[0], cols[1], cols[2]
                wav_path = wav_dir / f"{utt_id}.wav"
                if not wav_path.is_file():
                    skipped += 1
                    continue
                items.append({
                    "text": phones,
                    "audio_file": str(wav_path),
                    "speaker_name": "ruslan",
                    "root_path": str(wav_dir),
                })
        if skipped:
            print(f"  formatter skipped {skipped} rows")
        return items

    return formatter


def build_audio_config() -> BaseAudioConfig:
    """The audio config the lab 5 vocoder dictates. See the module docstring."""
    return BaseAudioConfig(
        sample_rate=SAMPLE_RATE, # Required for vocoder
        fft_size=1024,           # Required for vocoder
        win_length=1024,         # Required for vocoder
        hop_length=256,          # Required for vocoder
        num_mels=80,             # Required for vocoder
        mel_fmin=0.0,            # Required for vocoder
        mel_fmax=8000.0,         # Required for vocoder
        spec_gain=1.0,           # Required for vocoder 
        log_func="np.log",       # Required for vocoder
        signal_norm=False,       # Required for vocoder
        preemphasis=0.0,
        ref_level_db=20,
        do_trim_silence=False,   # Required for external durations
        trim_db=60.0,
    )



def preflight(tokenizer, samples, n: int = 200) -> None:
    """Abort before the first epoch if any phone is missing from the vocabulary.

    Without this the run starts, spends the data-loading pass mapping unknown phones to a
    single id, and trains to convergence on corrupted input. The failure has no symptom
    until synthesis sounds wrong hours later.
    """
    print("\n--- tokenizer pre-flight ---")
    print(f"vocabulary size: {tokenizer.characters.num_chars}")
    probe = samples[0]["text"]
    ids = tokenizer.text_to_ids(probe)
    print(f"sample phones  : {probe[:70]}...")
    print(f"sample ids     : {ids[:12]}... ({len(ids)} ids for {len(probe.split())} phones)")
    if len(ids) != len(probe.split()):
        print("  MISMATCH: id count != phone count. Tokens are being dropped.", file=sys.stderr)

    for item in samples[:n]:
        tokenizer.text_to_ids(item["text"])
    if tokenizer.not_found_characters:
        print(f"\n  ABORT: {len(tokenizer.not_found_characters)} tokens are not in the "
              f"vocabulary:", file=sys.stderr)
        for token in tokenizer.not_found_characters[:20]:
            codepoints = " ".join(f"U+{ord(c):04X}" for c in token)
            print(f"    {token!r}  {codepoints}", file=sys.stderr)
        print("\n  Most likely your phone metadata and phones_mfa.txt disagree, or one "
              "side is NFD and the other NFC.", file=sys.stderr)
        sys.exit(1)
    print(f"checked {min(n, len(samples))} utterances, no unknown tokens")
    print("--- pre-flight ok ---\n")




class EpochCheckpointTrainer(Trainer):
    """A Coqui Trainer with exactly one ordinary checkpoint per finished epoch.

    Keep Coqui's save_checkpoint() implementation: it records model, optimizer,
    scaler, global step and epoch and enforces save_n_checkpoints retention.
    """

    def _fit(self) -> None:
        self._restore_best_loss()
        # Some installed Coqui Trainer releases do not define restore_step/restore_epoch.
        # Our durable epoch_state.json is authoritative for completed epochs.
        if self.args.continue_path:
            run_dir = Path(self.output_path)
            state = json.loads((run_dir / 'epoch_state.json').read_text(encoding='utf-8'))
            start_epoch = int(state['completed_epochs'])
            restored_file = Path(str(self.args.restore_path or ''))
            expected_file = run_dir / str(state['checkpoint'])
            if not restored_file.is_file():
                raise RuntimeError(f'Coqui did not select a real checkpoint: {restored_file}')
            # Do not silently resume from an older best_model instead of the last epoch.
            import torch
            checkpoint = torch.load(str(restored_file), map_location='cpu', weights_only=False)
            restored_step = int(checkpoint['step'])
            expected_step = int(state['global_step'])
            del checkpoint
            if restored_step != expected_step:
                raise RuntimeError(
                    f'Resume mismatch: Coqui selected step {restored_step} from '
                    f'{restored_file.name}, last completed epoch has step {expected_step} '
                    f'in {expected_file.name}. No training performed.'
                )
            self.total_steps_done = expected_step
        else:
            start_epoch = 0
            self.total_steps_done = 0
        total_epochs = int(self.config.epochs)
        if start_epoch > total_epochs:
            raise RuntimeError(f"Completed {start_epoch} epochs, target is {total_epochs}")
        print(f"EPOCH PLAN: start={start_epoch} stop={total_epochs} "
              f"(completed={start_epoch}, remaining={total_epochs - start_epoch})", flush=True)
        if start_epoch == total_epochs:
            print("ALREADY COMPLETE: nothing to train", flush=True)
            return

        for epoch in range(start_epoch, total_epochs):
            if self.num_gpus > 1:
                dist.barrier()
            self.callbacks.on_epoch_start(self)
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, total_epochs, self.output_path)
            if not self.skip_train_epoch and not self.start_with_eval:
                self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
                self.test_run()
            self.c_logger.print_epoch_end(
                epoch,
                self.keep_avg_eval.avg_values if self.config.run_eval
                else self.keep_avg_train.avg_values,
            )
            if self.args.rank in (None, 0):
                self.save_best_model()
                self.save_checkpoint()
                self._write_epoch_record()
            self.callbacks.on_epoch_end(self)
            self.start_with_eval = False

    def fit(self) -> None:
        # Avoid Coqui Trainer.fit() deleting the entire experiment folder on an
        # exception (particularly problematic on Windows with open log handles).
        try:
            self._fit()
        finally:
            if self.args.rank in (None, 0):
                self.dashboard_logger.finish()

    def _write_epoch_record(self) -> None:
        run_path = Path(self.output_path)
        ordinary = sorted(
            run_path.glob('checkpoint_*.pth'),
            key=lambda path: int(path.stem.split('_')[-1]),
        )
        if not ordinary:
            raise RuntimeError("No checkpoint was saved at the end of the epoch")
        last = ordinary[-1]
        if not last.is_file() or last.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint is missing or empty: {last}")

        def loss(avg):
            if avg is None:
                return ''
            result = self._pick_target_avg_loss(avg)
            return '' if result is None else float(result)

        row = {
            'epoch': int(self.epochs_done) + 1,
            'global_step': int(self.total_steps_done),
            'train_loss': loss(self.keep_avg_train),
            'eval_loss': loss(self.keep_avg_eval),
            'checkpoint': last.name,
            'unix_time': round(time.time(), 3),
        }
        csv_path = run_path / 'epoch_history.csv'
        fields = list(row)
        exists = csv_path.exists() and csv_path.stat().st_size > 0
        with csv_path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())

        state = {
            'completed_epochs': row['epoch'],
            'last_completed_epoch_index': int(self.epochs_done),
            'global_step': row['global_step'],
            'checkpoint': last.name,
            'target_epochs': int(self.config.epochs),
        }
        state_path = run_path / 'epoch_state.json'
        temp_path = run_path / 'epoch_state.json.tmp'
        temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp_path, state_path)
        print(f"EPOCH SAVED: {row['epoch']}/{self.config.epochs} "
              f"step={row['global_step']} file={last.name} "
              f"retained={len(ordinary)}", flush=True)


def _batch_size_on_command_line() -> bool:
    for token in sys.argv[1:]:
        if token == "--batch-size" or token.startswith("--batch-size="):
            return True
    return False


def _point_dataset_at_local_files(trainer, args) -> None:
    datasets = getattr(trainer.config, "datasets", None) or []
    if not datasets:
        return
    dataset = datasets[0]
    meta = str(args.metadata)
    wavs = str(args.wavs)
    if isinstance(dataset, dict):
        dataset["meta_file_train"] = meta
        dataset["path"] = wavs
        if args.durations is not None:
            dataset["meta_file_attn_mask"] = str(args.durations)
    else:
        dataset.meta_file_train = meta
        dataset.path = wavs
        if args.durations is not None:
            dataset.meta_file_attn_mask = str(args.durations)
    print(f"dataset paths: metadata={meta} wavs={wavs}", flush=True)


def train_from_args(args) -> None:
    if args.save_n_checkpoints != 2:
        raise ValueError('This epoch-only training policy retains exactly 2 regular checkpoints')
    if args.epochs < 1 or (args.total_epochs is not None and args.total_epochs < 1):
        raise ValueError('Number of epochs must be positive')
    if args.continue_path:
        saved_config = args.continue_path / 'config.json'
        if not saved_config.is_file():
            raise FileNotFoundError(f'Cannot resume without {saved_config}')
        saved = json.loads(saved_config.read_text(encoding='utf-8'))
        saved_phones = (saved.get('characters') or {}).get('characters')
        current_phones = load_phone_inventory(args.phones)
        if saved_phones is not None and list(saved_phones) != current_phones:
            raise ValueError('Phone inventory differs from the saved run; refusing resume')
    output_path = args.continue_path if args.continue_path else args.output / args.run_name
    output_path.mkdir(parents=True, exist_ok=True)

    phones = load_phone_inventory(args.phones)
    print(f"phone inventory: {len(phones)} types from {args.phones}")

    dataset_config = BaseDatasetConfig(
        formatter="",  # unused: a callable goes to load_tts_samples below
        dataset_name="ruslan",
        meta_file_train=str(args.metadata),
        path=str(args.wavs),
        language="ru",
    )
    if args.durations is not None:
        # use_aligner=False makes the model read durations from here instead of learning
        # an alignment. The manifest is wav_path|npy_path, one line per utterance.
        dataset_config.meta_file_attn_mask = str(args.durations)
        print(f"durations: precomputed, from {args.durations}")
    else:
        print("durations: learned by the internal aligner")

    model_args = ForwardTTSArgs(
        use_pitch=args.pitch_energy,
        use_energy=args.pitch_energy,
        use_aligner=args.durations is None,
    )

    config = Fastspeech2Config(
        run_name=args.run_name,
        audio=build_audio_config(),
        batch_size=args.batch_size,
        eval_batch_size=max(2, args.batch_size // 2),
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        model_args=model_args,
        lr=args.lr,
        compute_input_seq_cache=True,
        compute_f0=args.pitch_energy,
        f0_cache_path=str(output_path / "f0_cache"),
        compute_energy=args.pitch_energy,
        energy_cache_path=str(output_path / "energy_cache"),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.total_epochs if args.total_epochs is not None else args.epochs,
        use_phonemes=False,
        text_cleaner="multicharacter_phoneme_cleaners",
        characters=CharactersConfig(
            characters_class=MULTICHAR_CLASS,
            characters=phones,
            punctuations="",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            is_unique=False,
            is_sorted=True,
        ),
        precompute_num_workers=args.precompute_num_workers,
        print_step=50,
        print_eval=False,
        mixed_precision=False,  # a 1080Ti is Pascal; fp16 buys nothing here
        max_seq_len=200,
        max_audio_len=int(args.max_audio_seconds * SAMPLE_RATE),
        output_path=str(output_path),
        save_step=2**60,  # disabled; saving is explicitly epoch-based
        save_checkpoints=False,  # disable built-in per-step checkpoint trigger
        save_on_interrupt=False,  # never write a mid-epoch checkpoint
        log_model_step=2**60,
        save_n_checkpoints=args.save_n_checkpoints,
        datasets=[dataset_config],
    )

    if args.pitch_energy:
        # The coqui defaults are pitch_fmin=1.0 and pitch_fmax=640.0. For a male speaker
        # at roughly 70-200 Hz that window is wrong in both directions: it admits octave
        # errors at the top and DC at the bottom, and it makes librosa.pyin crawl through
        # the f0 cache because the search range is nine times wider than it needs to be.
        config.audio.pitch_fmin = 50.0
        config.audio.pitch_fmax = 300.0
        print(f"f0 range: {config.audio.pitch_fmin}-{config.audio.pitch_fmax} Hz "
              f"(coqui's 1-640 default is wrong for this speaker)")

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    with args.metadata.open(encoding="utf-8-sig") as metadata_file:
        sample_count = sum(1 for line in metadata_file if len(line.rstrip("\n").split("|")) >= 3)
    if sample_count < 2:
        raise ValueError("At least two metadata records are required for train/eval splitting")
    config.eval_split_size = max(float(config.eval_split_size), 1.0 / sample_count + 1e-9)
    print(f"eval_split_size: {config.eval_split_size:.6f} for {sample_count} records")

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=make_formatter(args.wavs),
    )
    print(f"train: {len(train_samples)}   eval: {len(eval_samples)}")

    preflight(tokenizer, train_samples)

    # Test sentences go through this same tokenizer, so they must be phone strings rather
    # than Russian text. Taking them from the eval split guarantees every token is known.
    config.test_sentences = [item["text"] for item in eval_samples[:3]]

    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)
    original_characters = config.characters
    trainer = EpochCheckpointTrainer(
        TrainerArgs(continue_path=str(args.continue_path) if args.continue_path else None),
        config,
        str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        # This script parses its own flags. Coqui must not parse them again.
        parse_command_line_args=False,
    )
    # Coqui reloads the OLD config.json inside Trainer.__init__ on continue_path.
    # Reapply the requested epoch limit and the live phoneme inventory AFTER init.
    requested_epochs = args.total_epochs if args.total_epochs is not None else args.epochs
    trainer.config.epochs = requested_epochs
    trainer.config.characters = original_characters
    trainer.config.save_checkpoints = False
    trainer.config.save_on_interrupt = False
    trainer.config.save_step = 2**60
    trainer.config.log_model_step = 2**60
    trainer.config.save_n_checkpoints = args.save_n_checkpoints
    # A reloaded config.json overwrites the batch size and the dataset paths.
    # The doubled batch size is whatever this command passed; it is not multiplied again.
    if _batch_size_on_command_line() or getattr(args, "apply_batch_size", False):
        trainer.config.batch_size = args.batch_size
        trainer.config.eval_batch_size = max(2, args.batch_size // 2)
    if args.continue_path:
        _point_dataset_at_local_files(trainer, args)
    if args.continue_path:
        print(f"RESUMED: restore_path={trainer.args.restore_path}, "
              f"new_target_epochs={requested_epochs}", flush=True)
    if args.archive_first_epochs or args.archive_every_epochs:
        print("NOTE: archival flags ignored; only the latest two epoch checkpoints are kept",
              flush=True)
    print("SAVE POLICY: end of each epoch; keep latest "
          f"{args.save_n_checkpoints} ordinary checkpoints, plus one best model", flush=True)
    print(f"batch size: {trainer.config.batch_size} "
          f"(eval {trainer.config.eval_batch_size})", flush=True)
    trainer.fit()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actual_run(parent: Path) -> Path | None:
    dirs = {p.parent for p in parent.rglob("config.json")} if parent.exists() else set()
    if len(dirs) > 1:
        raise RuntimeError(f"Multiple Coqui runs found under {parent}: {sorted(dirs)}")
    return next(iter(dirs)) if dirs else None


def verify_run(run_dir: Path, expected_epochs: int) -> dict:
    state_file = run_dir / "epoch_state.json"
    history_file = run_dir / "epoch_history.csv"
    if not state_file.is_file() or not history_file.is_file():
        raise RuntimeError(f"Missing epoch_state.json or epoch_history.csv: {run_dir}")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    completed = int(state["completed_epochs"])
    if completed != expected_epochs:
        raise RuntimeError(f"Expected {expected_epochs} epochs, got {completed}: {run_dir}")
    with history_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    actual = [int(row["epoch"]) for row in rows]
    if actual != list(range(1, expected_epochs + 1)):
        raise RuntimeError(f"Nonsequential epoch history for {run_dir}: {actual}")
    checkpoints = list(run_dir.glob("checkpoint_*.pth"))
    if len(checkpoints) != min(2, completed):
        raise RuntimeError(f"Checkpoint retention failed for {run_dir}: {len(checkpoints)} files")
    last = run_dir / state["checkpoint"]
    if not last.is_file() or last.stat().st_size == 0:
        raise RuntimeError(f"Missing latest checkpoint: {last}")
    best = list(run_dir.glob("best_model*.pth"))
    if not best:
        raise RuntimeError(f"Missing best model: {run_dir}")
    return {
        "completed_epochs": completed,
        "global_step": int(state["global_step"]),
        "train_loss": rows[-1]["train_loss"],
        "eval_loss": rows[-1]["eval_loss"],
        "latest_checkpoint": last.name,
        "best_model": max(best, key=lambda path: path.stat().st_mtime).name,
        "checkpoint_count": len(checkpoints),
    }


def validate_metadata(metadata: Path, wavs: Path, *, check_all: bool = True) -> tuple[int, str]:
    n = 0
    missing = []
    digest = hashlib.sha256()
    with metadata.open("rb") as stream:
        for raw in stream:
            if len(raw.decode("utf-8-sig").rstrip("\r\n").split("|")) < 3:
                continue
            digest.update(raw)
            utt_id = raw.decode("utf-8-sig").split("|", 1)[0]
            n += 1
            if check_all and not (wavs / (utt_id + ".wav")).is_file() and len(missing) < 5:
                missing.append(utt_id)
    if n < 2:
        raise RuntimeError(f"Not enough utterances in {metadata}: {n}")
    if missing:
        raise RuntimeError(f"Missing WAV files for {metadata.name}: {missing}. Check --wavs")
    return n, digest.hexdigest()


def _apply_experiment_flags(ns, extra: list[str]) -> None:
    ns.pitch_energy = False
    ns.durations = None
    index = 0
    while index < len(extra):
        flag = extra[index]
        if flag == "--pitch-energy":
            ns.pitch_energy = True
        elif flag == "--durations":
            index += 1
            ns.durations = Path(extra[index])
        else:
            raise ValueError(f"Unknown experiment flag: {flag}")
        index += 1


def run_suite(args) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    names = list(dict.fromkeys((args.only or DEFAULT_CONFIGS) + (["mfa_dur_gt"] if args.include_mfa else [])))
    wavs = args.wavs if args.wavs.is_dir() else PROJECT_DIR / "data" / "RUSLAN" / "RUSLAN"
    wavs = wavs.resolve()
    phones = args.phones if args.phones.is_file() else PROJECT_DIR / "common" / "phones_mfa.txt"
    phones = phones.resolve()
    data_dir = PROJECT_DIR / "data"
    base_output = args.output if args.output != Path("runs") else LAB_DIR / "runs_epochonly_full"
    base_output = base_output.resolve()
    manifest = data_dir / "mfa_durations_lab4" / "manifest.csv"
    for path in (phones, wavs):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.dry_run:
        base_output.mkdir(parents=True, exist_ok=True)
    summary = []
    print(f"FULL DATASET: target={args.epochs}, resume={args.resume}, names={names}")
    print(f"WAV directory: {wavs}")
    print(f"Output: {base_output}")

    for name in names:
        metadata_name, lr, flags = EXPERIMENTS[name]
        metadata = data_dir / metadata_name
        if not metadata.is_file():
            raise FileNotFoundError(f"{name}: missing {metadata}. Prepare notebook Part 3 / Part 9")
        extra = list(flags)
        if name == "mfa_dur_gt":
            if not manifest.is_file():
                raise FileNotFoundError(f"MFA manifest missing: {manifest}")
            extra = ["--durations", str(manifest)]
        records, _ = validate_metadata(metadata, wavs)
        signature = {
            "experiment": name,
            "metadata": str(metadata.resolve()),
            "metadata_sha256": file_hash(metadata),
            "phones": str(phones),
            "phones_sha256": file_hash(phones),
            "wavs": str(wavs),
            "batch_size": args.batch_size,
            "learning_rate": lr,
            "flags": extra,
            "mfa_manifest_sha256": file_hash(manifest) if name == "mfa_dur_gt" else None,
        }
        parent = base_output / name
        manifest_file = parent / "full_run_manifest.json"
        run = actual_run(parent)
        continue_path = None
        total_epochs = None
        if args.resume:
            if run is None or not manifest_file.is_file():
                raise RuntimeError(f"{name}: no full run to resume; start with --epochs 5")
            previous = json.loads(manifest_file.read_text(encoding="utf-8"))
            if previous != signature:
                raise RuntimeError(f"{name}: metadata, phones, audio, hyperparameters or MFA manifest changed")
            state_path = run / "epoch_state.json"
            if not state_path.is_file():
                raise RuntimeError(f"{name}: no completed epoch checkpoint: {run}")
            completed = int(json.loads(state_path.read_text(encoding="utf-8"))["completed_epochs"])
            verify_run(run, completed)
            if completed > args.epochs:
                raise RuntimeError(f"{name}: already {completed} epochs; target {args.epochs} is smaller")
            if completed == args.epochs:
                print(f"SKIP {name}: already completed {completed} epochs", flush=True)
                summary.append({"experiment": name, "run_dir": str(run), **verify_run(run, completed)})
                continue
            print(f"RESUMING {name}: {completed} -> {args.epochs} total epochs ({records} metadata records)", flush=True)
            continue_path = run
            total_epochs = args.epochs
        else:
            if run is not None or manifest_file.exists():
                raise RuntimeError(f"{name}: existing run detected at {parent}. Use --resume; never overwrite it.")
            print(f"NEW {name}: 0 -> {args.epochs} epochs ({records} metadata records)", flush=True)
        one = argparse.Namespace(**vars(args))
        one.metadata = metadata
        one.wavs = wavs
        one.phones = phones
        one.output = parent
        one.run_name = name
        one.lr = lr
        one.batch_size = args.batch_size
        one.epochs = args.epochs
        one.total_epochs = total_epochs
        one.continue_path = continue_path
        one.num_loader_workers = 0
        one.num_eval_loader_workers = 0
        one.precompute_num_workers = 0
        one.save_every_epoch = True
        one.save_n_checkpoints = 2
        one.apply_batch_size = True
        _apply_experiment_flags(one, extra)
        if args.dry_run:
            print(f"PREVIEW: {name} metadata={metadata} lr={lr} epochs={args.epochs} "
                  f"continue={continue_path} flags={extra}", flush=True)
            continue
        parent.mkdir(parents=True, exist_ok=True)
        if not manifest_file.exists():
            manifest_file.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
        started = time.perf_counter()
        train_from_args(one)
        completed_run = actual_run(parent)
        if completed_run is None:
            raise RuntimeError(f"{name}: no Coqui output directory with config.json")
        stats = verify_run(completed_run, args.epochs)
        seconds_this_stage = round(time.perf_counter() - started, 1)
        record = {"experiment": name, "run_dir": str(completed_run),
                  "seconds_this_stage": seconds_this_stage, **stats}
        stages_path = parent / "training_stages.jsonl"
        with stages_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "experiment": name,
                "target_epochs": args.epochs,
                "seconds_this_stage": seconds_this_stage,
                "unix_time": time.time(),
            }, ensure_ascii=False) + "\n")
        summary.append(record)
        (parent / "full_run_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("PASS:", json.dumps(record, ensure_ascii=False), flush=True)
        summary_file = base_output / "training_summary.csv"
        with summary_file.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(summary)
    if args.dry_run:
        print("DRY RUN complete; no training started")
    else:
        print("FULL TRAINING STAGE FINISHED:", len(summary), "experiments", flush=True)


def main() -> None:
    """Build the config from the flags, check the data, and hand over to the trainer."""
    args = parse_args()
    if args.suite:
        if not _batch_size_on_command_line():
            args.batch_size = 128
        run_suite(args)
        return
    train_from_args(args)


if __name__ == "__main__":
    main()
