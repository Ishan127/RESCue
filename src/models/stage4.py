import torch
import torch.nn as nn
from typing import List

# Helper function is kept with the module that uses it.
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

class Stage4_GranularReasoner(nn.Module):
    """
    Conditional, hierarchical reasoner for finding parts within objects (for mRES).
    (RESCUE Stage 3)
    """
    def __init__(self, hidden_dim: int = 256, num_part_queries: int = 1):
        super().__init__()
        self.part_queries = nn.Parameter(torch.randn(1, num_part_queries, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def forward(self, parent_object_token: torch.Tensor, fused_tokens: torch.Tensor, padding_mask: torch.Tensor):
        queries = self.part_queries.repeat(fused_tokens.shape[0], 1, 1)
        primed_queries = queries + parent_object_token
        return self.decoder(tgt=primed_queries, memory=fused_tokens, memory_key_padding_mask=padding_mask)
