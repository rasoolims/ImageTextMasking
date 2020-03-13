import os
import tempfile
import unittest

import torch
import torch.utils.data as data_utils
from torchvision import transforms
from transformers import *

import attention
import dataset
import image_model
import image_text_model
import train


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
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")

        self.data = dataset.ImageTextDataset(data_idx_file=data_path, transform=transform, tokenizer=tokenizer)
        self.collator = dataset.ImageTextCollator(pad_idx=self.data.tokenizer.pad_token_id)
        self.text_encoder = AlbertModel.from_pretrained("albert-base-v1")
        self.image_model = image_model.init_net(embed_dim=768)
        self.loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_data(self):
        assert len(self.data) == 15
        assert len(self.data.texts[0]) < len(self.data.texts[-1])  # Make sure the data is sorted by length.

        for d in self.data:
            assert len(d) == 3

        assert len(self.data.label2idx) == 3

    def test_loader(self):
        loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)

        for d in loader:
            assert len(d) == 4
            hidden_reps, cls_head = self.text_encoder(d["texts"], attention_mask=d["pad_mask"])
            assert hidden_reps.size(0) == len(d["texts"])
            assert cls_head.size(0) == len(d["texts"])

    def test_image_model(self):
        loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)
        for d in loader:
            y = self.image_model(d["images"])
            assert y.size(1) == 49
            assert y.size(2) == 768
            assert d["images"].size(0) == y.size(0)

    def test_attention(self):
        text = torch.rand(4, 30, 128)
        image = torch.rand(4, 15, 128)
        mha = attention.MultiHeadedAttention(num_heads=2, d_model=128, dropout=0)
        a = mha(query=text, key=image, value=image)
        assert a.size() == text.size()

    def testImgTxtDecoderWithMasking(self):
        loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)
        model = image_text_model.ImageTextModel(text_encoder=self.text_encoder,
                                                image_encoder=self.image_model,
                                                tokenizer=self.data.tokenizer)
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print("Let's use", num_gpu, "GPUs!")
            model = torch.nn.DataParallel(model)
        model = model.to(self.device)

        for d in loader:
            result = model(device=self.device, data=d, mask_prob=0.5)  # choosing high mask prob for testing only
            output, masked_ids = result
            assert output.size(0) == masked_ids.size(0)
            assert output.size(1) == self.data.tokenizer.vocab_size
            assert masked_ids.dim() == 1
            break  # just testing the first case

    def testTrainer(self):
        loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)
        valid_loader = data_utils.DataLoader(self.data, batch_size=4, shuffle=False, collate_fn=self.collator)
        model = image_text_model.ImageTextModel(text_encoder=self.text_encoder, image_encoder=self.image_model,
                                                tokenizer=self.data.tokenizer)
        model = model.to(self.device)

        # choose high value for mask just for test
        trainer = train.Trainer(model=model, mask_prob=0.5)
        with tempfile.TemporaryDirectory() as tmpdirname:
            loss, best_valid_loss = trainer.train_epoch(data_iter=loader, valid_data_iter=valid_loader,
                                                        best_valid_loss=float("inf"),
                                                        saving_path=os.path.join(tmpdirname, "model"))
            assert float(loss.data) >= 0
            assert float(best_valid_loss) >= 0


if __name__ == '__main__':
    unittest.main()
