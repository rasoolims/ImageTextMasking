import os

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    def __init__(self, data_idx_file: str, transform, tokenizer):
        """

        :param data_idx_file: Each line is tab-separated. First: label, second: image path, others: tab-separated
            sentences.
        :param transform:
        :param tokenizer: BERT-style tokenizer.
        """
        self.transform = transform
        self.tokenizer = tokenizer
        IMG_EXTENSIONS = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'}

        init_labels = []
        init_images = []
        init_sentences = []
        batch_lens = []
        self.label2idx = {}
        self.idx2label = []

        with open(data_idx_file, "r") as reader:
            for line in reader:
                spl = line.strip().split("\t")
                label, image_path, sentences = spl[0], spl[1], spl[2:]
                if os.path.exists(image_path):
                    extension = image_path[image_path.rfind("."):]
                    if extension not in IMG_EXTENSIONS:
                        continue

                    for sen in sentences:
                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)
                            self.idx2label.append(label)
                        init_labels.append(self.label2idx[label])
                        tok_sen = self.tokenizer.encode(sen, add_special_tokens=True)
                        init_sentences.append(tok_sen)
                        batch_lens.append(len(tok_sen))
                        init_images.append(image_path)

        # Sorting the elements in the data based on batch length
        self.images = []
        self.texts = []
        self.labels = []
        len_ids = np.argsort(batch_lens)
        for len_id in len_ids:
            self.texts.append(init_sentences[len_id])
            self.images.append(init_images[len_id])
            self.labels.append(init_labels[len_id])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item: int):
        image = Image.open(self.images[item])
        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "text": torch.LongTensor(self.texts[item]),
                "label": torch.LongTensor([self.labels[item]])}


class ImageTextCollator(object):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        texts = [b["text"] for b in batch]
        labels = torch.stack([b["label"] for b in batch])

        padded_text = pad_sequence(texts, padding_value=self.pad_idx)

        return {"images": images, "texts": padded_text, "labels": labels}
