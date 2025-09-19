import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from typing import List, Dict

class Stage1_FusionModule(nn.Module):
    def __init__(self, hidden_dim: int = 256, vit_model_name='vit_small_patch16_224',clip_model_name='openai/clip-vit-base-patch16'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vit_encoder = timm.create_model(vit_model_name, pretrained=True)
        self.text_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.image_projector = nn.Linear(self.vit_encoder.embed_dim, hidden_dim)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.image_fusion_layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True) for _ in range(2)])
        self.text_fusion_layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True) for _ in range(2)])
        self.text_pos_embed = nn.Parameter(torch.randn(1, self.text_encoder.config.max_position_embeddings, hidden_dim))

    def forward(self, images: torch.Tensor, texts: list[str]):
        image_features = self.vit_encoder.forward_features(images)[:, 1:, :]
        text_inputs = self.text_tokenizer(texts, padding='max_length', return_tensors='pt', max_length=self.text_encoder.config.max_position_embeddings).to(images.device)
        text_features = self.text_encoder(**text_inputs).last_hidden_state
        image_features_proj = self.image_projector(image_features)
        text_features_proj = self.text_projector(text_features) + self.text_pos_embed
        updated_image_features, updated_text_features = image_features_proj, text_features_proj
        for img_layer, txt_layer in zip(self.image_fusion_layers, self.text_fusion_layers):
            temp_img = img_layer(tgt=updated_image_features, memory=updated_text_features, memory_key_padding_mask=text_inputs.attention_mask == 0)
            temp_txt = txt_layer(tgt=updated_text_features, memory=updated_image_features)
            updated_image_features, updated_text_features = temp_img, temp_txt
        return torch.cat([updated_image_features, updated_text_features], dim=1), text_inputs.attention_mask == 0

class Stage2_ObjectReasoner(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_queries: int = 10):
        super().__init__()
        self.num_queries = num_queries
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, fused_tokens: torch.Tensor, fused_tokens_padding_mask: torch.Tensor):
        queries = self.object_queries.repeat(fused_tokens.shape[0], 1, 1)
        return self.decoder(tgt=queries, memory=fused_tokens, memory_key_padding_mask=fused_tokens_padding_mask)

class HiRes_Core_Model(nn.Module):
    def __init__(self, image_size=224, patch_size=16):
        super().__init__()
        self.num_image_patches = (image_size // patch_size) ** 2
        self.stage1 = Stage1_FusionModule()
        self.num_text_tokens = self.stage1.text_encoder.config.max_position_embeddings
        self.stage2 = Stage2_ObjectReasoner(hidden_dim=self.stage1.hidden_dim)

    def forward(self, images: torch.Tensor, texts: list[str]):
        fused_tokens, text_padding_mask = self.stage1(images, texts)
        image_padding_mask = torch.zeros(fused_tokens.shape[0], self.num_image_patches, dtype=torch.bool, device=fused_tokens.device)
        full_padding_mask = torch.cat([image_padding_mask, text_padding_mask], dim=1)
        return self.stage2(fused_tokens, full_padding_mask)

# Additional model classes (ViTFeatureExtractor, PixelDecoderHighRes, Stage4_Mask2FormerDecoder, HiRes_Full_Model) should be added here following the notebook structure.
