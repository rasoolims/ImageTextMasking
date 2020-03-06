import os
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import transforms
from transformers import *

import dataset
import image_model
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
    def __init__(self, model: image_text_model.ImageTextModel, mask_prob: float = 0.15, optimizer=None):
        self.loss_compute = MaskLoss(optimizer=optimizer)
        self.model = model
        self.mask_prob = mask_prob

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print("Let's use", num_gpu, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def train_epoch(self, data_iter: data_utils.DataLoader):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens = 0, 0, 0

        for i, batch in enumerate(data_iter):
            predictions, target = self.model(device=self.device, data=batch, mask_prob=self.mask_prob)
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

    @staticmethod
    def train(data_path: str, num_epochs: int, mask_prob: float = 0.15):
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        img_model = image_model.init_net(embed_dim=768)  # todo as option!

        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
        text_encoder = AlbertModel.from_pretrained("albert-base-v1")

        train_data = dataset.ImageTextDataset(data_idx_file=data_path, transform=transform, tokenizer=tokenizer)
        collator = dataset.ImageTextCollator(pad_idx=train_data.tokenizer.pad_token_id)
        loader = data_utils.DataLoader(train_data, batch_size=4, shuffle=False, collate_fn=collator)

        model = image_text_model.ImageTextModel(text_encoder=text_encoder, image_encoder=img_model,
                                                tokenizer=tokenizer)  # todo other things as options in arg parser

        trainer = Trainer(model=model, mask_prob=mask_prob, optimizer=None)  # todo change optimizer

        for i in range(num_epochs):
            trainer.train_epoch(loader)


if __name__ == "__main__":
    data_path = os.path.abspath(sys.argv[1])
    num_epochs = int(sys.argv[2])  # todo: use arg parser
    mask_prob = float(sys.argv[3])

    Trainer.train(data_path=data_path, num_epochs=num_epochs, mask_prob=mask_prob)
