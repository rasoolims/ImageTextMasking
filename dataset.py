import logging
import os

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


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

        self.label2idx = {}
        self.idx2label = []
        data_dir = os.path.abspath(os.path.dirname(data_idx_file))

        # Sorting the elements in the data based on batch length
        self.images = []
        self.texts = []
        self.labels = []
        total_sentences = 0
        with open(data_idx_file, "r") as reader:
            for ln, line in enumerate(reader):
                spl = line.strip().split("\t")
                label, image_path, sentences = spl[0], spl[1], spl[2:]
                if not image_path.startswith("/"):
                    image_path = os.path.join(data_dir, image_path)

                if os.path.exists(image_path):
                    extension = image_path[image_path.rfind("."):]
                    if extension not in IMG_EXTENSIONS:
                        continue
                    the_sens = []

                    for sen in sentences:
                        tok_sen = self.tokenizer.encode(sen, add_special_tokens=True)
                        if len(tok_sen) <= 512:  # todo better splitting
                            if label not in self.label2idx:
                                self.label2idx[label] = len(self.label2idx)
                                self.idx2label.append(label)
                            the_sens.append(tok_sen)

                    if len(the_sens) > 0:
                        self.texts.append(the_sens)
                        total_sentences += len(the_sens)
                        self.labels.append(self.label2idx[label])
                        self.images.append(image_path)

                if (ln + 1) % 100000 == 0:
                    print(ln + 1, "-> loading number until now", len(self.images), "total sentences", total_sentences)

        print("loaded", len(self.images), "image/text pairs with", len(self.label2idx), "unique labels",
              "total sentences", total_sentences)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item: int):
        image = Image.open(self.images[item]).convert("RGB")  # make sure not to deal with rgba or grayscale images.
        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "text": self.texts[item],
                "label": torch.LongTensor([self.labels[item]])}


class ImageTextCollator(object):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        texts = [b["text"] for b in batch]
        text_to_image_map, max_len = [], 0
        text_lists = []
        for i, t in enumerate(texts):
            for l in t:
                max_len = max(len(l), max_len)
                text_to_image_map.append(i)
                text_lists.append(torch.LongTensor(l))

        labels = torch.stack([b["label"] for b in batch])

        padded_text = pad_sequence(text_lists, padding_value=self.pad_idx).T
        pad_mask = (padded_text == self.pad_idx)
        return {"images": images, "texts": padded_text, "labels": labels, "pad_mask": pad_mask,
                "text_image_map": text_to_image_map}
