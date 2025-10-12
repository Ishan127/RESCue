import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

class MultiScaleDeformableAttention(nn.Module):
    """
    A pure-PyTorch implementation of the Multi-Scale Deformable Attention module.
    This is a core component of Deformable DETR and is highly efficient.
    """
    def __init__(self, embed_dim, num_heads, num_levels, num_points, batch_first=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # Initialize sampling offsets to be a grid around the reference point
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * torch.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, value, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        B, Nq, C = query.shape
        B, Nv, _ = value.shape
        
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(B, Nv, self.num_heads, self.embed_dim // self.num_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(B, Nq, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(B, Nq, self.num_heads, self.num_levels, self.num_points)

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        output = self.ms_deform_attn_core(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

    @staticmethod
    def ms_deform_attn_core(value, spatial_shapes, sampling_locations, attention_weights):
        B, _, n_heads, D = value.shape
        _, Nq, n_heads, n_levels, n_points, _ = sampling_locations.shape
        value_list = value.split([H * W for H, W in spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid, (H, W) in enumerate(spatial_shapes):
            value_l_ = value_list[lid].flatten(2).transpose(1, 2).reshape(B * n_heads, D, H, W)
            sampling_grid_l_ = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Nq, n_levels * n_points)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(B, n_heads * D, Nq)
        return output.transpose(1, 2)
    
    
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

class Stage3_ObjectReasoner_Deformable(nn.Module):
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
