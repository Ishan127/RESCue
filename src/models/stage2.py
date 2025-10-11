import torch
import torch.nn as nn
import timm
from transformers import BertModel
from typing import Dict

class Stage2_FusionModule(nn.Module):
    """
    The entry point to the reasoning pipeline; fuses lightweight image and text features.
    (RESCUE Stage 1)
    """
    def __init__(self, hidden_dim: int = 256, vit_model_name: str = 'vit_base_patch16_224_in21k', text_model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vit_encoder = timm.create_model(vit_model_name, pretrained=True)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.image_projector = nn.Linear(self.vit_encoder.embed_dim, hidden_dim)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        fusion_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.image_fusion_layers = nn.ModuleList([fusion_layer for _ in range(2)])
        self.text_fusion_layers = nn.ModuleList([fusion_layer for _ in range(2)])
        self.text_pos_embed = nn.Parameter(torch.randn(1, self.text_encoder.config.max_position_embeddings, hidden_dim))

    def forward(self, images: torch.Tensor, text_inputs: Dict[str, torch.Tensor]):
        image_features = self.vit_encoder.forward_features(images)[:, 1:, :]
        text_features = self.text_encoder(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask']).last_hidden_state
        image_features_proj = self.image_projector(image_features)
        text_features_proj = self.text_projector(text_features) + self.text_pos_embed
        updated_image_features, updated_text_features = image_features_proj, text_features_proj
        for img_layer, txt_layer in zip(self.image_fusion_layers, self.text_fusion_layers):
            temp_img = img_layer(tgt=updated_image_features, memory=updated_text_features, memory_key_padding_mask=text_inputs["attention_mask"] == 0)
            temp_txt = txt_layer(tgt=updated_text_features, memory=updated_image_features)
            updated_image_features, updated_text_features = temp_img, temp_txt
        fused_tokens = torch.cat([updated_image_features, updated_text_features], dim=1)
        return fused_tokens, text_inputs["attention_mask"] == 0
