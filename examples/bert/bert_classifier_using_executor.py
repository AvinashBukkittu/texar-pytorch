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
"""Example of building a sentence classifier using Texar's Executor based on
pre-trained BERT model.
"""

import argparse
import functools
import importlib
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

import texar.torch as tx
from texar.torch.run import *
from texar.torch.modules import BERTClassifier

from utils import model_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-downstream", default="config_classifier",
    help="Configuration of the downstream part of the model")
parser.add_argument(
    '--pretrained-model-name', type=str, default='bert-base-uncased',
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    "--config-data", default="config_data", help="The dataset config.")
parser.add_argument(
    "--output-dir", default="output/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a model checkpoint (including bert modules) to restore from.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",
    help="Whether to run test on the test set.")
args = parser.parse_args()

config_data = importlib.import_module(args.config_data)
config_downstream = importlib.import_module(args.config_downstream)
config_downstream = {
    k: v for k, v in config_downstream.__dict__.items()
    if not k.startswith('__')}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.root.setLevel(logging.INFO)


class ModelWrapper(nn.Module):
    def __init__(self, model: BERTClassifier):
        super().__init__()
        self.model = model

    def _compute_loss(self, logits, labels):
        r"""Compute loss.
        """
        if self.model.is_binary:
            loss = F.binary_cross_entropy(
                logits.view(-1), labels.view(-1), reduction='mean')
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.model.num_classes),
                labels.view(-1), reduction='mean')
        return loss

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        labels = batch["label_ids"]

        input_length = (1 - (input_ids == 0).int()).sum(dim=1)

        logits, _ = self.model(input_ids, input_length, segment_ids)

        loss = self._compute_loss(logits, labels)

        return {"loss": loss}

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        predictions = self.model(encoder_input=batch.source,
                                 beam_width=self.beam_width)
        if self.beam_width == 1:
            decoded_ids = predictions[0].sample_id
        else:
            decoded_ids = predictions["sample_id"][:, :, 0]
        return {"preds": decoded_ids}


def main():
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)

    # Loads data
    num_train_data = config_data.num_train_data

    # Builds BERT
    model = tx.modules.BERTClassifier(
        pretrained_model_name=args.pretrained_model_name,
        hparams=config_downstream)
    model = ModelWrapper(model=model)

    num_train_steps = int(num_train_data / config_data.train_batch_size *
                          config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)

    # Builds learning rate decay scheduler
    static_lr = 2e-5

    vars_with_decay = []
    vars_without_decay = []
    for name, param in model.named_parameters():
        if 'layer_norm' in name or name.endswith('bias'):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    opt_params = [{
        'params': vars_with_decay,
        'weight_decay': 0.01,
    }, {
        'params': vars_without_decay,
        'weight_decay': 0.0,
    }]
    optim = tx.core.BertAdam(
        opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, functools.partial(model_utils.get_lr_multiplier,
                                 total_steps=num_train_steps,
                                 warmup_steps=num_warmup_steps))

    train_dataset = tx.data.RecordData(hparams=config_data.train_hparam,
                                       device=device)
    eval_dataset = tx.data.RecordData(hparams=config_data.eval_hparam,
                                      device=device)
    test_dataset = tx.data.RecordData(hparams=config_data.test_hparam,
                                      device=device)

    iterator = tx.data.DataIterator(
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
    )

    batching_strategy = tx.data.TokenCountBatchingStrategy(
        max_tokens=config_data.max_batch_tokens)

    output_dir = Path(args.output_dir)
    executor = Executor(
        model=model,
        train_data=train_dataset,
        valid_data=eval_dataset,
        test_data=test_dataset,
        batching_strategy=batching_strategy,
        device=device,
        optimizer=optim,
        lr_scheduler=scheduler,
        log_destination=[sys.stdout, output_dir / "log.txt"],
        log_every=cond.iteration(config_data.display_steps),
        validate_every=None,
        stop_training_on=cond.epoch(config_data.max_train_epoch),
        train_metrics=[
            ("loss", metric.RunningAverage(1)),  # only show current loss
            ("lr", metric.LR(optim))],
        log_format="{time} : Epoch {epoch:2d} @ {iteration:6d}it "
                   "({progress}%, {speed}), lr = {lr:.3e}, loss = {loss:.3f}",
        checkpoint_dir=args.output_dir,
        save_every=cond.validation(better=True),
        max_to_keep=1,
        show_live_progress=True,
    )

    executor.train()


if __name__ == "__main__":
    main()
