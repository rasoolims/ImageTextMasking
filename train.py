import time

import torch.nn as nn
import torch.utils.data as data_utils

import image_text_model


class MaskLoss:
    def __init__(self, optimizer=None):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def __call__(self, prediction, gold_standard, norm):
        loss = self.criterion(prediction.contiguous().view(-1, prediction.size(-1)),
                              gold_standard.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        return loss.data * norm


class Trainer:
    def __init__(self, model: image_text_model.ImageTextModel, vocab_size: int, padding_idx: int,
                 mask_prob: float = 0.15, optimizer=None):
        self.loss_compute = MaskLoss(optimizer=optimizer)
        self.model = model
        self.mask_prob = mask_prob

    def train_epoch(self, device, data_iter: data_utils.DataLoader):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens = 0, 0, 0

        for i, batch in enumerate(data_iter):
            predictions, target = self.model(device=device, data=batch, mask_prob=self.mask_prob)
            ntokens = target.size(0)

            if ntokens == 0:  # Nothing to predict!
                continue

            loss = self.loss_compute(predictions, target, ntokens)
            total_loss += loss
            total_tokens += ntokens
            tokens += ntokens

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                      (i + 1, loss / ntokens, tokens / elapsed))
                start, tokens = time.time(), 0
        return total_loss / total_tokens
