import torch
import torch.nn as nn
import timm

class Stage1_SAM_Like_Encoder(nn.Module):
    """
    The heavyweight, frozen image encoder that creates the 'master canvas'.
    (RESCUE Stage 0)
    """
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
