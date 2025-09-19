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

    def forward(self, images: torch.Tensor, text_inputs):
        image_features = self.vit_encoder.forward_features(images)[:, 1:, :]
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

    def forward(self, images: torch.Tensor, text_inputs):
        fused_tokens, text_padding_mask = self.stage1(images, text_inputs)
        image_padding_mask = torch.zeros(fused_tokens.shape[0], self.num_image_patches, dtype=torch.bool, device=fused_tokens.device)
        full_padding_mask = torch.cat([image_padding_mask, text_padding_mask], dim=1)
        return self.stage2(fused_tokens, full_padding_mask)


class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224_in21k', feature_indices=(2, 5, 8, 11)):
        super().__init__()
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.feature_indices = feature_indices
        self.patch_size = self.vit.patch_embed.patch_size[0]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        H_patch, W_patch = H // self.patch_size, W // self.patch_size
        x = self.vit.patch_embed(x)
        x = torch.cat((self.vit.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        features = {}
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.feature_indices:
                feature_map = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H_patch, W_patch)
                features[f"scale_{i}"] = feature_map
        return features

class UpsampleBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)

class PixelDecoderHighRes(nn.Module):
    def __init__(self, input_dims: Dict[str, int], output_dim: int = 256, image_size: int = 224, vit_patch: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.patch = vit_patch
        self.input_proj = nn.ModuleDict({
            name: nn.Conv2d(in_dim, output_dim, kernel_size=1)
            for name, in_dim in input_dims.items()
        })
        self.fuse = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU()
        )
        grid_size = image_size // vit_patch
        assert image_size % vit_patch == 0, "image_size must be divisible by vit_patch"
        steps = int(np.round(np.log2(image_size / grid_size)))
        self.ups = nn.ModuleList([UpsampleBlock(output_dim) for _ in range(steps)])
        self.refine_full = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU()
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = sorted(features.keys(), key=lambda k: int(k.split("_")[-1]))
        proj = []
        target_hw = None
        for k in keys:
            x = features[k]
            x = self.input_proj[k](x)
            if target_hw is None:
                target_hw = x.shape[-2:]
                proj.append(x)
            else:
                proj.append(F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False))
        fused = torch.stack(proj, dim=0).sum(0)
        fused = self.fuse(fused)
        y = fused
        for up in self.ups:
            y = up(y)
        y = self.refine_full(y)
        return y

class Stage4_Mask2FormerDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_queries: int = 10):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True)
        self.query_refiner = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.mask_embed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, object_tokens: torch.Tensor, pixel_embeddings: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_embeddings.shape
        pixel_embeddings_flat = pixel_embeddings.flatten(2).permute(0, 2, 1)
        refined_tokens = self.query_refiner(tgt=object_tokens, memory=pixel_embeddings_flat)
        mask_embeddings = self.mask_embed_head(refined_tokens)
        mask_logits = (mask_embeddings @ pixel_embeddings.flatten(2)) / np.sqrt(pixel_embeddings.shape[1])
        return mask_logits.view(B, -1, H, W)

class HiRes_Full_Model(nn.Module):
    def __init__(self, image_size=224, patch_size=16, hidden_dim=256, num_queries=1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_extractor = ViTFeatureExtractor(vit_model_name='vit_base_patch16_224_in21k', feature_indices=(2,5,8,11))
        self.reasoning_core = HiRes_Core_Model(image_size=image_size, patch_size=patch_size)
        feature_dims = {f"scale_{i}": 768 for i in self.feature_extractor.feature_indices}
        self.pixel_decoder = PixelDecoderHighRes(
            input_dims=feature_dims,
            output_dim=hidden_dim,
            image_size=image_size,
            vit_patch=self.feature_extractor.patch_size
        )
        self.mask_decoder = Stage4_Mask2FormerDecoder(hidden_dim=hidden_dim, num_queries=num_queries)

    def forward(self, images: torch.Tensor, text_inputs) -> Dict[str, torch.Tensor]:
        multi_scale_features = self.feature_extractor(images)
        pixel_embeddings = self.pixel_decoder(multi_scale_features)
        object_tokens = self.reasoning_core(images, text_inputs)
        predicted_masks = self.mask_decoder(object_tokens, pixel_embeddings)
        return {"pred_masks": predicted_masks}
