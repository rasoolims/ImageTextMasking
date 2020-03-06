import time
from optparse import OptionParser

import torch
import torch.utils.data as data_utils
from torchvision import transforms
from transformers import *

import dataset
import image_model
import image_text_model


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """
    Got it from
    https://github.com/pytorch/fairseq/blob/46b773a393c423f653887c382e4d55e69627454d/fairseq/criterions/label_smoothed_cross_entropy.py
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


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
        self.criterion = label_smoothed_nll_loss
        self.optimizer = optimizer

    def __call__(self, prediction, gold_standard, norm):
        loss = self.criterion(prediction.contiguous().view(-1, prediction.size(-1)),
                              gold_standard.contiguous().view(-1), epsilon=0.1)

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
                       torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

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
    def train(options):
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

        train_data = dataset.ImageTextDataset(data_idx_file=options.data_path, transform=transform, tokenizer=tokenizer)
        collator = dataset.ImageTextCollator(pad_idx=train_data.tokenizer.pad_token_id)
        loader = data_utils.DataLoader(train_data, batch_size=4, shuffle=False, collate_fn=collator)

        model = image_text_model.ImageTextModel(text_encoder=text_encoder, image_encoder=img_model,
                                                tokenizer=tokenizer,
                                                d_model=options.d_model,
                                                dropout=options.dropout,
                                                d_ff=options.d_ff,
                                                num_layers=options.num_layers,
                                                num_heads=options.num_heads)
        trainer = Trainer(model=model, mask_prob=options.mask_prob, optimizer=Trainer.get_std_opt(model))

        for i in range(options.num_epochs):
            trainer.train_epoch(loader)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--data", dest="data_path", help="Path to the data folder", metavar="FILE", default=None)
    parser.add_option("--epoch", dest="num_epochs", help="Number of training epochs", type="int", default=25)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.15)
    parser.add_option("--embed", dest="d_model", help="Embedding of contextual word vectors", type="int", default=768)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--layer", dest="num_layers", help="Number of Layers in cross-attention", type="int", default=2)
    parser.add_option("--heads", dest="num_heads", help="Number of attention heads", type="int", default=8)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    Trainer.train(options=options)
