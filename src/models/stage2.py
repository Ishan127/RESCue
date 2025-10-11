import torch
import torch.nn as nn
import timm
from transformers import BertModel
from typing import Dict

class EnhancedFusionLayer(nn.Module):
    """
    A single, structured fusion layer inspired by GroundingDINO's "Feature Enhancer".
    It performs a sequence of self-attention followed by cross-attention for deep refinement.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        # Self-Attention Layers
        self.img_self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.txt_self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_img1 = nn.LayerNorm(d_model)
        self.norm_txt1 = nn.LayerNorm(d_model)

        # Cross-Attention Layers
        self.img_cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.txt_cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_img2 = nn.LayerNorm(d_model)
        self.norm_txt2 = nn.LayerNorm(d_model)

        # Feed-Forward Networks
        self.ffn_img = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ffn_txt = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm_img3 = nn.LayerNorm(d_model)
        self.norm_txt3 = nn.LayerNorm(d_model)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor, txt_padding_mask: torch.Tensor):
        # 1. Introspection (Self-Attention)
        # Image self-attention
        img_sa = self.img_self_attn(img_feat, img_feat, img_feat)[0]
        img_feat = self.norm_img1(img_feat + img_sa)
        
        # Text self-attention
        txt_sa = self.txt_self_attn(txt_feat, txt_feat, txt_feat, key_padding_mask=txt_padding_mask)[0]
        txt_feat = self.norm_txt1(txt_feat + txt_sa)

        # 2. Dialogue (Cross-Attention)
        # Image queries text
        img_ca = self.img_cross_attn(query=img_feat, key=txt_feat, value=txt_feat, key_padding_mask=txt_padding_mask)[0]
        img_feat = self.norm_img2(img_feat + img_ca)

        # Text queries image
        txt_ca = self.txt_cross_attn(query=txt_feat, key=img_feat, value=img_feat)[0]
        txt_feat = self.norm_txt2(txt_feat + txt_ca)
        
        # 3. Feed-Forward Networks
        img_ffn = self.ffn_img(img_feat)
        img_feat = self.norm_img3(img_feat + img_ffn)
        
        txt_ffn = self.ffn_txt(txt_feat)
        txt_feat = self.norm_txt3(txt_feat + txt_ffn)
        
        return img_feat, txt_feat

class Stage2_FusionModule(nn.Module):
    """
    The entry point to the reasoning pipeline; fuses lightweight image and text features.
    (RESCUE Stage 1 - UPGRADED with Structured Fusion)
    """
    def __init__(self, hidden_dim: int = 256, vit_model_name: str = 'vit_base_patch16_224_in21k', text_model_name: str = 'bert-base-uncased', fusion_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vit_encoder = timm.create_model(vit_model_name, pretrained=True)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.image_projector = nn.Linear(self.vit_encoder.embed_dim, hidden_dim)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.text_pos_embed = nn.Parameter(torch.randn(1, self.text_encoder.config.max_position_embeddings, hidden_dim))

        # <<< FIX APPLIED HERE: Using the new EnhancedFusionLayer >>>
        self.fusion_layers = nn.ModuleList([
            EnhancedFusionLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4) 
            for _ in range(fusion_layers)
        ])

    def forward(self, images: torch.Tensor, text_inputs: Dict[str, torch.Tensor]):
        # --- 1. Unimodal Encoding & Projection ---
        image_features = self.vit_encoder.forward_features(images)[:, 1:, :]
        text_features = self.text_encoder(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask']).last_hidden_state
        image_features_proj = self.image_projector(image_features)
        text_features_proj = self.text_projector(text_features) + self.text_pos_embed
        
        updated_image_features = image_features_proj
        updated_text_features = text_features_proj
        txt_padding_mask = text_inputs["attention_mask"] == 0

        # --- 2. Iterative Structured Fusion ---
        for layer in self.fusion_layers:
            updated_image_features, updated_text_features = layer(
                updated_image_features, 
                updated_text_features,
                txt_padding_mask
            )

        # --- 3. Final Concatenation ---
        fused_tokens = torch.cat([updated_image_features, updated_text_features], dim=1)
        return fused_tokens, txt_padding_mask

