from typing import Dict

from attention import *


class ImageTextModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, mask_id: int, d_model: int = 768, dropout: float = 0.1,
                 d_ff: int = 2048,
                 num_layers: int = 2,
                 num_heads: int = 8):
        super(ImageTextModel, self).__init__()

        attention = MultiHeadedAttention(num_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.mask_id = mask_id
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        self.decoder = Decoder(
            layer=DecoderLayer(size=d_model, self_attn=copy.deepcopy(attention), src_attn=copy.deepcopy(attention),
                               feed_forward=ff, dropout=dropout), N=num_layers)

    def forward(self, data: Dict[str, torch.Tensor], mask_prob: float = 0.0):
        """
        :param data: A minibatch as dictionary that has transformed image and tokenized text as long tensors.
        :return:
        """
        image_hidden = self.image_encoder(data["images"])

        texts, pads = data["texts"], data["pad_mask"]
        mask, masked_ids = None, None
        if mask_prob > 0:
            assert 0 < mask_prob < 1
            mask = torch.empty(texts.size()).uniform_(0, 1) < mask_prob
            mask[0] = False
            mask[pads] = False  # We should not mask pads.
            masked_ids = texts[mask]
            texts[mask] = self.mask_id

        text_hidden, text_cls_head = self.text_encoder(texts, attention_mask=pads)
        decoder_output = self.decoder(text=text_hidden, image=image_hidden, text_mask=pads)
        return decoder_output, mask, masked_ids
