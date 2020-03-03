import os
import unittest

import torch.utils.data as data_utils
from torchvision import transforms
from transformers import *

import dataset
import image_model


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
        data_path = os.path.join(path_dir_name, "small_data/labels.txt")
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        self.data = dataset.ImageTextDataset(data_idx_file=data_path, transform=transform, tokenizer=tokenizer)
        self.collator = dataset.ImageTextCollator(pad_idx=self.data.tokenizer.pad_token_id)

    def test_data(self):
        assert len(self.data) == 29
        assert len(self.data.texts[0]) < len(self.data.texts[-1])  # Make sure the data is sorted by length.

        for d in self.data:
            assert len(d) == 3

        assert len(self.data.label2idx) == 3

    def test_loader(self):
        loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)
        roberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        for d in loader:
            assert len(d) == 4
            hidden_reps, cls_head = roberta_model(d["texts"], attention_mask=d["pad_mask"])
            assert hidden_reps.size(0) <= 4
            assert cls_head.size(0) <= 4

    def test_image_model(self):
        model = image_model.init_net()

        loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)
        for d in loader:
            y = model(d["images"])
            assert y.size(1) == y.size(2) == 7
            assert y.size(3) == 2048
            assert d["images"].size(0) == y.size(0)


if __name__ == '__main__':
    unittest.main()
