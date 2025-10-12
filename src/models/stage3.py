import torch
import torch.nn as nn

class Stage3_ObjectReasoner(nn.Module):
    """
    Finds primary objects (for RES/gRES) from the fused context.
    (RESCUE Stage 2)
    """
    def __init__(self, hidden_dim: int = 256, num_queries: int = 10):
        super().__init__()
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, fused_tokens: torch.Tensor, padding_mask: torch.Tensor):
        queries = self.object_queries.repeat(fused_tokens.shape[0], 1, 1)
        return self.decoder(tgt=queries, memory=fused_tokens, memory_key_padding_mask=padding_mask)
