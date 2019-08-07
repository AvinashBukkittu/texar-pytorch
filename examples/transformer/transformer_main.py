# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model.
"""

import argparse
import functools
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
import texar.torch as tx
from texar.torch.run import *

from model import Transformer
import utils.data_utils as data_utils
import utils.utils as utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-model", type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    "--config-data", type=str, default="config_iwslt15",
    help="The dataset config.")
parser.add_argument(
    "--run-mode", type=str, default="train_and_evaluate",
    help="Either train_and_evaluate or evaluate or test.")
parser.add_argument(
    "--output-dir", type=str, default="./outputs/",
    help="Path to save the trained model and logs.")
parser.add_argument(
    "--load-checkpoint", action="store_true", default=False,
    help="If specified, will load the pre-trained checkpoint from output_dir.")

args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

make_deterministic(config_model.random_seed)


class ModelWrapper(nn.Module):
    def __init__(self, model: Transformer, beam_width: int):
        super().__init__()
        self.model = model
        self.beam_width = beam_width

    def forward(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        loss = self.model(encoder_input=batch.source,
                          decoder_input=batch.target_input,
                          labels=batch.target_output)
        return {"loss": loss}

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        predictions = self.model(encoder_input=batch.source,
                                 beam_width=self.beam_width)
        if self.beam_width == 1:
            decoded_ids = predictions[0].sample_id
        else:
            decoded_ids = predictions["sample_id"][:, :, 0]
        return {"preds": decoded_ids}


def spm_decode(tokens: List[str]) -> List[str]:
    spm_bos_token = "▁"
    words = []
    pieces = []
    for t in tokens:
        if t[0] == spm_bos_token:
            if len(pieces) > 0:
                words.append(''.join(pieces))
            pieces = [t[1:]]
        else:
            pieces.append(t)
    if len(pieces) > 0:
        words.append(''.join(pieces))
    return words


class FileBLEUMetric(metric.SimpleMetric[List[int], float]):
    def __init__(self, vocab: tx.data.Vocab,
                 file_path: Optional[str] = None):
        super().__init__(pred_name="preds", label_name="target_output")
        self.vocab = vocab
        self.file_path = file_path

    @property
    def metric_name(self) -> str:
        return "BLEU"

    def _to_str(self, tokens: List[int]) -> str:
        pos = next((idx for idx, x in enumerate(tokens)
                    if x == self.vocab.eos_token_id), -1)
        if pos != -1:
            tokens = tokens[:pos]
        vocab_map = self.vocab.id_to_token_map_py

        words = spm_decode([vocab_map[t] for t in tokens])
        sentence = ' '.join(words)
        return sentence

    def value(self) -> float:
        if len(self.predicted) == 0:
            return 0.0
        path = self.file_path or tempfile.mktemp()
        hypotheses, references = [], []
        for hyp, ref in zip(self.predicted, self.labels):
            hypotheses.append(self._to_str(hyp))
            references.append(self._to_str(ref))
        hyp_file, ref_file = tx.utils.write_paired_text(
            hypotheses, references,
            path, mode="s", src_fname_suffix="hyp", tgt_fname_suffix="ref")
        bleu = tx.evals.file_bleu(ref_file, hyp_file, case_sensitive=True)
        return bleu


class BLEUWrapper(metric.BLEU):
    def __init__(self, vocab: tx.data.Vocab):
        super().__init__(pred_name="preds", label_name="target_output")
        self.vocab = vocab

    @property
    def metric_name(self) -> str:
        return "BLEU"

    def _to_str(self, tokens: List[int]) -> str:
        pos = next((idx for idx, x in enumerate(tokens)
                    if x == self.vocab.eos_token_id), -1)
        if pos != -1:
            tokens = tokens[:pos]
        vocab_map = self.vocab.id_to_token_map_py

        words = [vocab_map[t] for t in tokens]
        sentence = ' '.join(words)
        return sentence

    def add(self, predicted, labels) -> None:
        predicted = [self._to_str(s) for s in predicted]
        labels = [self._to_str(s) for s in labels]
        super().add(predicted, labels)


def main():
    """Entry point.
    """
    # Load data
    vocab = tx.data.Vocab(config_data.vocab_file)
    data_hparams = {
        # "batch_size" is ignored for train since we use dynamic batching
        "batch_size": config_data.test_batch_size,
        "bos_id": vocab.bos_token_id,
        "eos_id": vocab.eos_token_id,
    }
    datasets = {
        split: data_utils.Seq2SeqData(
            os.path.join(
                config_data.input_dir,
                f"{config_data.filename_prefix}{split}.npy"),
            hparams={
                **data_hparams,
                "shuffle": args.run_mode == "train_and_evaluate",
            })
        for split in ["train", "valid", "test"]
    }
    print(f"Training data size: {len(datasets['train'])}")
    batching_strategy = data_utils.CustomBatchingStrategy(
        config_data.max_batch_tokens)

    # Create model and optimizer
    model = Transformer(config_model, config_data, vocab)
    model = ModelWrapper(model, config_model.beam_width)

    lr_config = config_model.lr_config
    if lr_config["learning_rate_schedule"] == "static":
        init_lr = lr_config["static_lr"]
        scheduler_lambda = lambda x: 1.0
    else:
        init_lr = lr_config["lr_constant"]
        scheduler_lambda = functools.partial(
            utils.get_lr_multiplier, warmup_steps=lr_config["warmup_steps"])
    optim = torch.optim.Adam(
        model.parameters(), lr=init_lr, betas=(0.9, 0.997), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)

    output_dir = Path(args.output_dir)
    executor = Executor(
        model=model,
        train_data=datasets["train"],
        valid_data=datasets["valid"],
        test_data=datasets["test"],
        batching_strategy=batching_strategy,
        optimizer=optim,
        lr_scheduler=scheduler,
        log_destination=[sys.stdout, output_dir / "log.txt"],
        log_every=cond.iteration(config_data.display_steps),
        validate_every=[cond.iteration(config_data.eval_steps), cond.epoch(1)],
        stop_training_on=cond.epoch(config_data.max_train_epoch),
        train_metrics=[
            ("loss", metric.RunningAverage(1)),  # only show current loss
            ("lr", metric.LR(optim))],
        log_format="{time} : Epoch {epoch:2d} @ {iteration:6d}it "
                   "({progress}%, {speed}), lr = {lr:.3e}, loss = {loss:.3f}",
        valid_metrics=BLEUWrapper(vocab),
        test_metrics=FileBLEUMetric(vocab, output_dir / "test.output"),
        validate_mode='predict',
        checkpoint_dir=args.output_dir,
        save_every=cond.validation(better=True),
        max_to_keep=1,
        show_live_progress=["train", "valid"],
    )
    if args.run_mode == "train_and_evaluate":
        executor.write_log("Begin running with train_and_evaluate mode")
        if args.load_checkpoint:
            load_path = executor.load(allow_failure=True)
            if load_path is not None:
                executor.test({"valid": datasets["valid"]})

        executor.train()

    elif args.run_mode in ["evaluate", "test"]:
        executor.write_log(f"Begin running with {args.run_mode} mode")
        executor.load(load_training_state=False)
        split = "test" if args.run_mode == "test" else "valid"
        executor.test({split: datasets[split]})

    else:
        raise ValueError(f"Unknown mode: {args.run_mode}")


if __name__ == "__main__":
    main()
