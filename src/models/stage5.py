import torch
import torch.nn as nn

class Stage5_MaskDecoder(nn.Module):
    """
    The final rendering pipeline; decodes a target token into a mask using the master canvas.
    (RESCUE Stage 4)
    """
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
