from typing import Dict, List, Optional, Tuple
from torch import nn, Tensor
import copy
import json
import math
import numpy as np
import random
import torch
import torch.nn.functional as F




class Config(object):
    dropout = 0.1
    attention_dropout = 0.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embed
        self.self_attn = Attention(self.embed_dim,
                                   config.n_head,
                                   dropout=config.attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = F.gelu

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, attn_mask, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x

        x = self.self_attn(query=x,
                           key=x,
                           attn_mask = attn_mask,
                           key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        x = self.self_attn_layer_norm(x)

        residual = x

        x = self.fc2(self.activation_fn(self.fc1(x)))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
  
        x = self.final_layer_norm(x)

        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)

        return x

class AutoModel(nn.Module):

    def __init__(self, padding_idx = 256, embed_dim = 256, num_layer = 4, device = "cuda"):
        super().__init__()
        self.padding_idx = padding_idx
        self.device = device
        self.config = Config(
            max_position_embeddings=1000,
            n_layer=num_layer,
            n_head=8,
            n_embed = embed_dim,
            pad_token_id=padding_idx,
            ffn_dim=embed_dim,
        )

        self.dropout = self.config.dropout
        self.embed_dim = embed_dim
        self.hidden_size = embed_dim
        
        self.max_source_positions = self.config.max_position_embeddings
   
        self.pos_embed = LearnedPositionalEmbedding(self.config.max_position_embeddings, self.embed_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(self.config) for _ in range(self.config.n_layer)])
        self.layernorm_embedding = LayerNorm(self.embed_dim) 
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def forward(self, source):
        """
        Compute the logits for the given source.

        Args:
            source: The input data.
            **unused: Additional unused arguments.

        Returns:
            logits: The computed logits.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################

        bsz, seq_len, _ = source.shape

        encoder_padding_mask, causal_mask = _prepare_decoder_inputs(
            self.config,
            decoder_input_ids=source
        )
    
        x = source + self.pos_embed(source)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for encoder_layer in self.layers:
            x = encoder_layer(x, attn_mask = causal_mask, encoder_padding_mask = encoder_padding_mask)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x



def make_padding_mask(input_ids, padding_idx=0):
    """True for pad tokens"""
    padding_mask = input_ids[:, :, 0].eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _prepare_decoder_inputs(config,
                            decoder_input_ids):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    bsz, tgt_len, _ = decoder_input_ids.size()

    decoder_padding_mask = make_padding_mask(decoder_input_ids)

    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=float,
                         device=decoder_input_ids.device)
    return decoder_padding_mask, causal_mask

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        super().__init__(num_embeddings,
                         embedding_dim)

    def forward(self, input_ids):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        assert seq_len<1000
        positions = torch.arange(seq_len,
                                 dtype=torch.long,
                                 device=self.weight.device)
        return super().forward(positions)



class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads,
                                        self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute the attention output. You need to apply key_padding_mask and attn_mask before softmax operation.

        Args:
            query (torch.Tensor): The input query tensor, shape (seq_len, batch_size, embed_dim).
            key (Optional[torch.Tensor]): The input key tensor, shape (seq_len, batch_size, embed_dim).
                                         If None, it's assumed to be the same as the query tensor.
            key_padding_mask (Optional[torch.Tensor]): The key padding mask tensor, shape (batch_size, seq_len).
                                                      Default: None
            attn_mask (Optional[torch.Tensor]): The attention mask tensor, shape (seq_len, seq_len).
                                               Default: None

        Returns:
            attn_output (torch.Tensor): The attention output tensor, shape (seq_len, batch_size, embed_dim).

        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # flag = (query.shape == key.shape)
        # if not flag:
        #     print('eq:', query.shape, key.shape)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(key)

        q_seq_len, bsz, embed_dim = query.shape

        k_seq_len, _, _ = key.shape

        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            attn_mask = attn_mask.unsqueeze(0)
 

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = self._shape(q, q_seq_len, bsz)
        k = self._shape(k, k_seq_len, bsz)
        v = self._shape(v, k_seq_len, bsz)


        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, k_seq_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, k_seq_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.logical_or(key_padding_mask)
                # attn_mask = attn_mask.float()
                # attn_mask.masked_fill_(key_padding_mask, float("-inf"))

        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        q *= self.scaling

        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        attn_output_weights = F.softmax(attn, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)
    
        attn_output = attn_output.transpose(0, 1).contiguous().view(q_seq_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        return attn_output


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):

    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

# class LearnedPositionalEmbedding(nn.Embedding):
#     """
#     This module learns positional embeddings up to a fixed maximum size.
#     Padding ids are ignored by either offsetting based on padding_idx
#     or by setting padding_idx to None and ensuring that the appropriate
#     position ids are passed to the forward function.
#     """

#     def __init__(self, num_embeddings: int, embedding_dim: int,
#                  padding_idx: int, offset):
#         # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
#         # and adjust num_embeddings appropriately. Other models dont have this hack
#         self.offset = offset
#         assert padding_idx is not None
#         num_embeddings += offset
#         super().__init__(num_embeddings,
#                          embedding_dim)

#     def forward(self, input_ids):
#         """Input is expected to be of size [bsz x seqlen]."""
#         bsz, seq_len = input_ids.shape[:2]

#         positions = torch.arange(seq_len,
#                                  dtype=torch.long,
#                                  device=self.weight.device)
#         return super().forward(positions + self.offset)
