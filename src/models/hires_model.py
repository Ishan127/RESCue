import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import CLIPTokenizer, CLIPTextModel, BertModel
import warnings
from typing import List, Dict
import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message="The `pad_to_max_length` argument is deprecated.*", category=FutureWarning)

# ======================================================================================
# HELPER FOR CONDITIONAL LOGIC (STAGE 3)
# ======================================================================================

GRANULAR_CUES = [
    'hat', 'sleeve', 'collar', 'pocket', 'logo', 'emblem', 'strap',
    'wheel', 'tire', 'headlight', 'taillight', 'mirror', 'door', 'handle', 'window', 'bumper',
    'ear', 'eye', 'nose', 'mouth', 'tail', 'paw', 'leg', 'wing', 'beak',
    'top', 'bottom', 'left', 'right', 'corner', 'edge', 'side', 'front', 'back'
]

def detect_granular_cue(texts: List[str]) -> torch.Tensor:
    """Simulates linguistic cue detection to determine if Stage 3 should be activated."""
    batch_has_cue = [any(cue in text.lower() for cue in GRANULAR_CUES) for text in texts]
    return torch.tensor(batch_has_cue, dtype=torch.bool)

# ======================================================================================
# INDIVIDUAL STAGE MODULES
# ======================================================================================

class Stage0_SAM_Like_Encoder(nn.Module):
    """The heavyweight, frozen image encoder that creates the 'master canvas'."""
    def __init__(self, vit_model_name: str = 'vit_huge_patch14_224_in21k', image_size: int = 224, freeze: bool = True):
        super().__init__()
        self.vit_encoder = timm.create_model(vit_model_name, pretrained=True)
        self.embed_dim = self.vit_encoder.embed_dim
        self.patch_size = self.vit_encoder.patch_embed.patch_size[0]
        self.grid_size = image_size // self.patch_size
        if freeze:
            for param in self.vit_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.vit_encoder.eval()
        patch_embeddings = self.vit_encoder.forward_features(images)
        image_patch_tokens = patch_embeddings[:, 1:, :]
        b, n, c = image_patch_tokens.shape
        master_canvas = image_patch_tokens.reshape(b, self.grid_size, self.grid_size, c).permute(0, 3, 1, 2)
        return master_canvas

class Stage1_FusionModule(nn.Module):
    """The entry point to the reasoning pipeline; fuses lightweight image and text features."""
    def __init__(self, hidden_dim: int = 256, vit_model_name: str = 'vit_base_patch16_224_in21k', text_model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vit_encoder = timm.create_model(vit_model_name, pretrained=True)
        # Note: Tokenizer is now expected to be used outside the model
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.image_projector = nn.Linear(self.vit_encoder.embed_dim, hidden_dim)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        fusion_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.image_fusion_layers = nn.ModuleList([fusion_layer for _ in range(2)])
        self.text_fusion_layers = nn.ModuleList([fusion_layer for _ in range(2)])
        self.text_pos_embed = nn.Parameter(torch.randn(1, self.text_encoder.config.max_position_embeddings, hidden_dim))

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        image_features = self.vit_encoder.forward_features(images)[:, 1:, :]
        
        # <<< FIX APPLIED HERE: Pass tensors directly, not a dictionary >>>
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        image_features_proj = self.image_projector(image_features)
        text_features_proj = self.text_projector(text_features) + self.text_pos_embed
        updated_image_features, updated_text_features = image_features_proj, text_features_proj
        for img_layer, txt_layer in zip(self.image_fusion_layers, self.text_fusion_layers):
            # <<< FIX APPLIED HERE: Use the attention_mask tensor directly >>>
            temp_img = img_layer(tgt=updated_image_features, memory=updated_text_features, memory_key_padding_mask=attention_mask == 0)
            temp_txt = txt_layer(tgt=updated_text_features, memory=updated_image_features)
            updated_image_features, updated_text_features = temp_img, temp_txt
        fused_tokens = torch.cat([updated_image_features, updated_text_features], dim=1)
        return fused_tokens, attention_mask == 0

class Stage2_ObjectReasoner(nn.Module):
    """Finds primary objects (for RES/gRES) from the fused context."""
    def __init__(self, hidden_dim: int = 256, num_queries: int = 10):
        super().__init__()
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, fused_tokens: torch.Tensor, padding_mask: torch.Tensor):
        queries = self.object_queries.repeat(fused_tokens.shape[0], 1, 1)
        return self.decoder(tgt=queries, memory=fused_tokens, memory_key_padding_mask=padding_mask)

class Stage3_GranularReasoner(nn.Module):
    """Conditional, hierarchical reasoner for finding parts within objects (for mRES)."""
    def __init__(self, hidden_dim: int = 256, num_part_queries: int = 1):
        super().__init__()
        self.part_queries = nn.Parameter(torch.randn(1, num_part_queries, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def forward(self, parent_object_token: torch.Tensor, fused_tokens: torch.Tensor, padding_mask: torch.Tensor):
        queries = self.part_queries.repeat(fused_tokens.shape[0], 1, 1)
        primed_queries = queries + parent_object_token
        return self.decoder(tgt=primed_queries, memory=fused_tokens, memory_key_padding_mask=padding_mask)

class Stage4_MaskDecoder(nn.Module):
    """The final rendering pipeline; decodes a target token into a mask using the master canvas."""
    def __init__(self, reasoning_dim: int = 256, hires_embedding_dim: int = 1280):
        super().__init__()
        self.input_proj = nn.Conv2d(hires_embedding_dim, reasoning_dim, kernel_size=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=reasoning_dim, nhead=8, dim_feedforward=reasoning_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.mask_embed_head = nn.Sequential(nn.Linear(reasoning_dim, reasoning_dim), nn.ReLU(), nn.Linear(reasoning_dim, reasoning_dim))

    def forward(self, high_res_image_embeddings: torch.Tensor, final_target_tokens: torch.Tensor):
        projected_embeddings = self.input_proj(high_res_image_embeddings)
        b, c, h, w = projected_embeddings.shape
        pos_embed = self._generate_positional_encodings(b, h, w, c, device=projected_embeddings.device)
        projected_embeddings_flat = projected_embeddings.flatten(2).permute(0, 2, 1)
        refined_tokens = self.decoder(tgt=final_target_tokens, memory=projected_embeddings_flat + pos_embed)
        mask_embeddings = self.mask_embed_head(refined_tokens)
        mask_logits_flat = mask_embeddings @ projected_embeddings_flat.permute(0, 2, 1)
        return mask_logits_flat.view(b, -1, h, w)

    def _generate_positional_encodings(self, b, h, w, c, device):
        pos_x = torch.arange(w, device=device, dtype=torch.float32).view(1, -1, 1).expand(h, -1, -1)
        pos_y = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1, 1).expand(-1, w, -1)
        div_term = torch.exp(torch.arange(0, c, 2, device=device, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / c))
        pos_enc = torch.zeros(h, w, c, device=device)
        pos_enc[:, :, 0::2] = torch.sin(pos_x * div_term)
        pos_enc[:, :, 1::2] = torch.cos(pos_y * div_term)
        return pos_enc.flatten(0,1).unsqueeze(0).expand(b, -1, -1)

# ======================================================================================
# THE COMPLETE, MODULAR RESCUE MODEL
# ======================================================================================

class RESCUE_Model(nn.Module):
    """The full, integrated RESCUE architecture, designed for repo consistency."""
    def __init__(self, image_size=224, hidden_dim=256, num_object_queries=10, **kwargs):
        super().__init__()
        self.image_size = image_size
        
        # --- Initialize all stages ---
        self.stage0_encoder = Stage0_SAM_Like_Encoder(image_size=image_size)
        self.stage1_fusion = Stage1_FusionModule(hidden_dim=hidden_dim)
        self.stage2_reasoner = Stage2_ObjectReasoner(hidden_dim=hidden_dim, num_queries=num_object_queries)
        self.stage3_reasoner = Stage3_GranularReasoner(hidden_dim=hidden_dim)
        self.stage4_decoder = Stage4_MaskDecoder(reasoning_dim=hidden_dim, hires_embedding_dim=self.stage0_encoder.embed_dim)

        # Helper attributes
        self.num_image_patches = (image_size // self.stage1_fusion.vit_encoder.patch_embed.patch_size[0]) ** 2
        
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, run_stage3_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # --- PHASE I: FEATURE EXTRACTION ---
        high_res_image_embeddings = self.stage0_encoder(images)
        # <<< FIX APPLIED HERE: Pass tensors directly >>>
        fused_tokens, text_padding_mask = self.stage1_fusion(images, input_ids, attention_mask)
        full_padding_mask = self._create_full_padding_mask(fused_tokens, text_padding_mask)

        # --- PHASE II: REASONING PIPELINE ---
        object_centric_tokens = self.stage2_reasoner(fused_tokens, full_padding_mask)
        parent_tokens = object_centric_tokens[:, 0:1, :] 
        part_centric_tokens = self.stage3_reasoner(parent_object_token=parent_tokens, fused_tokens=fused_tokens, padding_mask=full_padding_mask)
        
        # --- CONDITIONAL SELECTION ---
        selection_mask = run_stage3_mask.view(-1, 1, 1).expand_as(object_centric_tokens)
        mres_targets = object_centric_tokens.clone()
        mres_targets[:, 0:1, :] = part_centric_tokens
        final_target_tokens = torch.where(selection_mask, mres_targets, object_centric_tokens)

        # --- PHASE III: RENDERING ---
        final_mask_logits = self.stage4_decoder(high_res_image_embeddings, final_target_tokens)
        final_masks = F.interpolate(final_mask_logits, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return {"pred_masks": final_masks}

    def _create_full_padding_mask(self, fused_tokens, text_padding_mask):
        image_padding_mask = torch.zeros(
            fused_tokens.shape[0], self.num_image_patches, 
            dtype=torch.bool, device=fused_tokens.device
        )
        return torch.cat([image_padding_mask, text_padding_mask], dim=1)
