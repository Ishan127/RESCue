import torch
import torch.nn as nn
from torchvision.ops import MultiScaleDeformableAttention

class DeformableTransformerDecoderLayer(nn.Module):
    """A decoder layer that uses Deformable Attention for efficient cross-attention."""
    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, n_levels=1, n_points=4):
        super().__init__()
        # Standard self-attention for queries to interact with each other
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Deformable cross-attention for queries to attend to the image-text context
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_heads, n_levels, n_points, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # Standard Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query, reference_points, memory, memory_spatial_shapes, level_start_index, memory_key_padding_mask):
        # 1. Self-Attention
        q_sa = self.self_attn(query, query, query)[0]
        query = self.norm1(query + q_sa)

        # 2. Deformable Cross-Attention
        q_ca = self.cross_attn(
            query=query,
            value=memory,
            reference_points=reference_points,
            spatial_shapes=memory_spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=memory_key_padding_mask
        )
        query = self.norm2(query + q_ca)
        
        # 3. Feed-Forward Network
        q_ffn = self.ffn(query)
        query = self.norm3(query + q_ffn)
        
        return query

class Stage2_ObjectReasoner_Deformable(nn.Module):
    """
    An upgraded Stage 2 that uses Deformable Attention.
    """
    def __init__(self, hidden_dim: int = 256, num_queries: int = 10, decoder_layers: int = 2):
        super().__init__()
        self.num_queries = num_queries
        
        # Still has learnable content queries
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # And now has learnable spatial queries (reference points)
        self.reference_points = nn.Parameter(torch.rand(1, num_queries, 2)) # x, y coordinates

        # The decoder is now a stack of our new Deformable layers
        deformable_layer = DeformableTransformerDecoderLayer(d_model=hidden_dim)
        self.decoder_layers = nn.ModuleList([deformable_layer for _ in range(decoder_layers)])

    def forward(self, fused_tokens: torch.Tensor, fused_tokens_padding_mask: torch.Tensor, spatial_shapes: torch.Tensor):
        batch_size = fused_tokens.shape[0]
        
        # Prepare queries and reference points for the batch
        queries = self.object_queries.repeat(batch_size, 1, 1)
        ref_points = self.reference_points.repeat(batch_size, 1, 1)
        
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), 
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))

        # Pass through the deformable decoder layers
        for layer in self.decoder_layers:
            queries = layer(
                query=queries,
                reference_points=ref_points,
                memory=fused_tokens,
                memory_spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_key_padding_mask=fused_tokens_padding_mask
            )
        
        return queries
