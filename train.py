import time
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import transforms
from transformers import *

import dataset
import image_model
import image_text_model


class NoamOpt:
    "Optim wrapper that implements rate."

    """
    from https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


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

    @staticmethod
    def get_std_opt(model):
        return NoamOpt(model.d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

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

        trainer = Trainer(model=model, mask_prob=mask_prob, optimizer=Trainer.get_std_opt(model))

        for i in range(num_epochs):
            trainer.train_epoch(loader)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--data", dest="data_path", help="Path to the data folder", metavar="FILE", default=None)
    parser.add_option("--epoch", dest="num_epochs", help="Number of training epochs", type="int", default=25)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.15)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    Trainer.train(data_path=options.data_path, num_epochs=options.num_epochs, mask_prob=options.mask_prob)
