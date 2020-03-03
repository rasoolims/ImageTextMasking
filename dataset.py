import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    def __init__(self, data_folder: str, transform, tokenizer):
        """

        :param data_folder: Has 2 subfolders: img, and txt. Each has file names starting from 0. It also
        has a labels.txt file that has exactly n={number of img and txt files} lines.
        :param transform:
        :param tokenizer: BERT-style tokenizer.
        """
        self.transform = transform
        self.tokenizer = tokenizer

        init_multi_texts = glob.glob(os.path.join(data_folder, "txt", "*.txt"))  # each has many lines
        IMG_EXTENSIONS = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'}
        image_folder = os.path.join(data_folder, "img")

        with open(os.path.join(data_folder, "labels.txt"), "r") as reader:
            init_labels = reader.read().strip().split("\n")
            self.label2idx = {}
            self.idx2label = []
            for label in init_labels:
                if label not in self.label2idx:
                    self.label2idx[label] = len(self.label2idx)
                    self.idx2label.append(label)

        multi_texts = {}
        for t in init_multi_texts:
            with open(t, "r") as reader:
                text_raw_list = reader.read().strip().split("\n")
            text_tensor_list = [self.tokenizer.encode(text, add_special_tokens=True) for text in text_raw_list]
            file_number = t[t.rfind("/") + 1:t.rfind(".")]
            # the output should be list of string
            multi_texts[file_number] = text_tensor_list

        batch_lens, texts, images, labels = [], [], [], []
        for init_image in os.listdir(image_folder):
            init_image = os.path.join(image_folder, init_image)
            extension = init_image[init_image.rfind("."):]
            if extension not in IMG_EXTENSIONS:
                continue
            file_number = init_image[init_image.rfind("/") + 1:init_image.rfind(".")]
            for text_sentence in multi_texts[file_number]:
                images.append(init_image)
                texts.append(text_sentence)
                batch_lens.append(len(text_sentence))
                labels.append(self.label2idx[init_labels[int(file_number)]])

        # Sorting the elements in the data based on batch length
        self.images = []
        self.texts = []
        self.labels = []
        len_ids = np.argsort(batch_lens)
        for len_id in len_ids:
            self.texts.append(texts[len_id])
            self.images.append(images[len_id])
            self.labels.append(labels[len_id])

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
