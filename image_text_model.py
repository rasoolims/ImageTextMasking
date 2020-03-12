from typing import Dict

from attention import *


class ImageTextModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, tokenizer, d_model: int = 768,
                 dropout: float = 0.1,
                 d_ff: int = 2048,
                 num_layers: int = 2,
                 num_heads: int = 8):
        super(ImageTextModel, self).__init__()
        self.tokenizer = tokenizer

        attention = MultiHeadedAttention(num_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.d_model = d_model

        self.decoder = Decoder(
            layer=DecoderLayer(size=d_model, src_attn=attention,
                               feed_forward=ff, dropout=dropout), N=num_layers)
        self.output_layer = nn.Linear(d_model, self.tokenizer.vocab_size, )

    def forward(self, device, data: Dict[str, torch.Tensor], mask_prob: float = 0.15):
        """
        :param data: A minibatch as dictionary that has transformed image and tokenized text as long tensors.
        :return:
        """
        images = data["images"].to(device)
        texts = data["texts"].to(device)
        pads = data["pad_mask"].to(device)
        
        image_hidden = self.image_encoder(images)

        assert 0 < mask_prob < 1
        mask = torch.empty(texts.size()).uniform_(0, 1) < mask_prob
        mask = mask.to(device)
        mask[0] = False

        mask[pads] = False  # We should not mask pads.
        masked_ids = texts[mask]
        texts[mask] = self.tokenizer.mask_token_id

        text_hidden, text_cls_head = self.text_encoder(texts, attention_mask=pads)
        decoder_output = self.decoder(text=text_hidden, image=image_hidden, text_mask=pads)

        output_predictions = F.log_softmax(self.output_layer(decoder_output[mask]), dim=1)
        return output_predictions, masked_ids
