import os
import unittest

import torch.utils.data as data_utils
from torchvision import transforms
from transformers import *

import dataset


class TestDataSet(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(path_dir_name, "small_data")
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        self.data = dataset.ImageTextDataset(data_folder=data_dir, transform=transform, tokenizer=tokenizer)

    def test_data(self):
        assert len(self.data) == 32
        assert len(self.data.texts[0]) < len(self.data.texts[-1])  # Make sure the data is sorted by length.

        for d in self.data:
            assert len(d) == 3

    def test_loader(self):
        collator = dataset.ImageTextCollator(pad_idx=self.data.tokenizer.pad_token_id)
        loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=collator)

        for d in loader:
            assert len(d) == 3


if __name__ == '__main__':
    unittest.main()
