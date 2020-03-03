import os
import unittest

from torchvision import transforms
from transformers import *

import dataset


class TestDataSet(unittest.TestCase):
    def test_data(self):
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

        data = dataset.ImageTextDataset(data_folder=data_dir, transform=transform, tokenizer=tokenizer)
        assert len(data) == 32
        assert len(data.texts[0]) < len(data.texts[-1]) # Make sure the data is sorted by length.

        for d in data:
            assert len(d) == 3

if __name__ == '__main__':
    unittest.main()