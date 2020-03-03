import glob
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    def __init__(self, data_folder: str, transform, tokenizer):
        """

        :param data_folder: Has 2 subfolders: img, and txt. Each has file names starting from 0. It also
        has a labels.txt file that has extactly n={number of img and txt files} lines.
        :param transform:
        :param tokenizer: BERT-style tokenizer.
        """
        self.transform = transform
        init_multi_texts = glob.glob(os.path.join(data_folder, "txt", "*.txt"))  # each has many lines
        IMG_EXTENSIONS = set(['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])
        image_folder = os.path.join(data_folder, "img")

        with open(os.path.join(data_folder, "labels.txt"), "r") as reader:
            init_labels = reader.read().strip().split("\n")

        multi_texts = {}
        for t in init_multi_texts:
            with open(t, "r") as reader:
                text_raw_list = reader.read().strip().split("\n")
            text_tensor_list = [tokenizer.encode(text, add_special_tokens=True) for text in text_raw_list]
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
                labels.append(init_labels[int(file_number)])

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

        return {"image": image, "text": self.texts[item], "label": self.labels[item]}
