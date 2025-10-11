import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

# Use relative imports for a clean repository structure
from .stage1 import Stage1_SAM_Like_Encoder
from .stage2 import Stage2_FusionModule
from .stage3 import Stage3_ObjectReasoner
from .stage4 import Stage4_GranularReasoner, detect_granular_cue
from .stage5 import Stage5_MaskDecoder

class RESCUE_Model(nn.Module):
    """
    The full, integrated RESCUE architecture, designed for repo consistency.
    This class orchestrates the entire pipeline from input to final mask prediction.
    """
    def __init__(self, image_size=224, hidden_dim=256, num_object_queries=10, **kwargs):
        super().__init__()
        self.image_size = image_size
        
        # --- Initialize all stages by importing them ---
        self.stage0_encoder = Stage1_SAM_Like_Encoder(image_size=image_size)
        self.stage1_fusion = Stage2_FusionModule(hidden_dim=hidden_dim)
        self.stage2_reasoner = Stage3_ObjectReasoner(hidden_dim=hidden_dim, num_queries=num_object_queries)
        self.stage3_reasoner = Stage4_GranularReasoner(hidden_dim=hidden_dim)
        self.stage4_decoder = Stage5_MaskDecoder(reasoning_dim=hidden_dim, hires_embedding_dim=self.stage0_encoder.embed_dim)

        # Helper attribute
        self.num_image_patches = (image_size // self.stage1_fusion.vit_encoder.patch_embed.patch_size[0]) ** 2
        
    def forward(self, images: torch.Tensor, texts: List[str], text_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        The full forward pass implementing the conditional reasoning workflow.
        """
        
        # --- PHASE I: FEATURE EXTRACTION ---
        high_res_image_embeddings = self.stage0_encoder(images)
        fused_tokens, text_padding_mask = self.stage1_fusion(images, text_inputs)
        full_padding_mask = self._create_full_padding_mask(fused_tokens, text_padding_mask)

        # --- PHASE II: REASONING PIPELINE (UNCONDITIONAL EXECUTION) ---
        object_centric_tokens = self.stage2_reasoner(fused_tokens, full_padding_mask)
        parent_tokens = object_centric_tokens[:, 0:1, :] 
        part_centric_tokens = self.stage3_reasoner(
            parent_object_token=parent_tokens,
            fused_tokens=fused_tokens,
            padding_mask=full_padding_mask
        )
        
        # --- CONDITIONAL SELECTION OF FINAL TARGETS ---
        run_stage3_mask = detect_granular_cue(texts).to(images.device)
        selection_mask = run_stage3_mask.view(-1, 1, 1).expand_as(object_centric_tokens)
        
        mres_targets = object_centric_tokens.clone()
        mres_targets[:, 0:1, :] = part_centric_tokens

        final_target_tokens = torch.where(selection_mask, mres_targets, object_centric_tokens)

        # --- PHASE III: RENDERING ---
        final_mask_logits = self.stage4_decoder(high_res_image_embeddings, final_target_tokens)
        
        final_masks = F.interpolate(
            final_mask_logits,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        return {"pred_masks": final_masks}

    def _create_full_padding_mask(self, fused_tokens, text_padding_mask):
        image_padding_mask = torch.zeros(
            fused_tokens.shape[0], self.num_image_patches, 
            dtype=torch.bool, device=fused_tokens.device
        )
        return torch.cat([image_padding_mask, text_padding_mask], dim=1)

