from typing import Dict, List, Optional, Tuple
from torch import nn, Tensor
import copy
import json
import numpy as np
import random
import torch
import torch.nn.functional as F
import math


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):

    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

class Config(object):
    dropout = 0.0
    scale_embedding = None
    static_position_embeddings = False
    extra_pos_embeddings = 0
    normalize_before = False
    normalize_embedding = True
    add_final_layer_norm = False
    init_std = 0.02

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class VertexDecoder(nn.Module):    #xzc write by myself for autoregressive generation
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.embed_scale = math.sqrt(
            config.n_embed) if config.scale_embedding else 1.0
        self.layers = nn.ModuleList([
            VertexDecoderLayer(config) for _ in range(config.n_layer)
        ])
        self.layernorm_embedding = LayerNorm(
            config.n_embed) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(
            config.n_embed) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        decoder_padding_mask,
        decoder_causal_mask,
    ):
        """
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_padding_mask: for ignoring pad tokens
            decoder_causal_mask: mask the future tokens
        """
        x = self.layernorm_embedding(input_ids)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)

        for _, decoder_layer in enumerate(self.layers):
            x = decoder_layer(
                x,
                decoder_padding_mask=decoder_padding_mask,
                causal_mask=decoder_causal_mask,
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)

        return x


class VertexDecoderLayer(nn.Module):  # decoderlayer for autoregressive generation

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embed

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.n_head,
            dropout=0,
        )
        self.self_attn2 = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.n_head,
            dropout=0,
        )
        self.dropout = config.dropout
        self.activation_fn = F.relu
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.n_head,
            dropout=0,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        causal_mask=None,
        decoder_padding_mask=None,
    ):
        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x = self.self_attn2(
            query=x,
            key=x,
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class VertexModel(nn.Module):

    def __init__(self, embedding_dim=256, num_layers=4, hidden_size=256):
        super().__init__()
        self.padding_idx = 258
        self.config = Config(
            n_embed=embedding_dim,
            n_layer=num_layers,
            n_head=8,
            pad_token_id=self.padding_idx,
            ffn_dim= hidden_size,
        )
        print(self.config)
        self.decoder = VertexDecoder(self.config)
        self.out_proj = nn.Linear(embedding_dim, 2**8+3)   # 0-255, <eos>=256, start=257, padding=258
        
        # 3 embeddings below
        self.value_embedding = torch.nn.Embedding(2**8 + 3,
            embedding_dim, padding_idx=258)
        self.coord_embedding = torch.nn.Embedding(4, embedding_dim)
        self.pos_embedding = torch.nn.Embedding(1000, embedding_dim)

    def logits(self, source):
        """
        Compute the logits for the given source.

        Args:
            source: The input data.
        Returns:
            logits: The computed logits.
        """

        source = source.to('cuda')
        bs = source.shape[0]
        seqlen = source.shape[-1]    # seqlen = 3k+1
        coord_tokens = (torch.arange(seqlen) % 3).reshape(1, seqlen).repeat(bs, 1).cuda()   # [bs, seq]
        coord_tokens =  coord_tokens.masked_fill(source >= 256, 3).cuda()
        pos_tokens = ((torch.arange(seqlen, device='cuda') + 2) // 3).reshape(1, seqlen).repeat(bs, 1)

        value_embed = self.value_embedding(source.cuda())
        coord_embed = self.coord_embedding(coord_tokens)
        pos_embed = self.pos_embedding(pos_tokens)
        # merge
        x = value_embed + coord_embed + pos_embed

        _, decoder_padding_mask, causal_mask = _prepare_decoder_inputs(
            self.config,
            input_ids=None,
            decoder_input_ids=source,
            causal_mask_dtype=torch.float32,
        )
        hidden = self.decoder(x, decoder_padding_mask, causal_mask)
        logits = self.out_proj(hidden)
        return logits

    def get_loss(self, source, target, reduce=True):
        source = source.cuda()
        target = target.cuda()
        logits = self.logits(source)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs.cuda(),
            target.view(-1).cuda(),
            ignore_index=258,
            reduction="mean" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, prefix, file_path):
        # start=257, eos=256, pad=258
        # prefix: [int, int, ...]   known coordinates
        prefix = torch.tensor([257]+prefix, device='cuda').reshape(1, -1)
        while True:
            logits = self.logits(prefix)    # [bs=1, seq, 259]
            idx = -1
            while True:
                idx = torch.multinomial(logits[0,-1, :257].softmax(-1), 1)
                assert idx <= 256
                if idx == 256:
                    if prefix.shape[1] % 3 != 1:
                        continue
                    else:
                        prefix = torch.cat([prefix, torch.tensor([[idx]], device='cuda')], axis=1)
                        break
                else:   # 0-255
                    if prefix.shape[1] < 4:
                        prefix = torch.cat([prefix, torch.tensor([[idx]], device='cuda')], axis=1)
                        break
                    if prefix.shape[1] % 3 == 1:    # z-coordinate non-decrease
                        if idx < prefix[0,-3]:
                            continue
                    if prefix.shape[1] % 3 == 2:    # y-coordinate non-decrease if z-coord equal
                        if prefix[0,-1] == prefix[0,-4] and idx < prefix[0,-3]:
                            continue
                    if prefix.shape[1] % 3 == 0:    # x-coord strictly increase if z-coord equal && y-coord equal
                        if prefix[0,-1] == prefix[0,-4] and prefix[0,-2]==prefix[0,-5] and idx <= prefix[0,-3]:
                            continue
                    prefix = torch.cat([prefix, torch.tensor([[idx]], device='cuda')], axis=1)
                    break
            if idx == 256:
                break
            if prefix.shape[1] > 290:
                break
        vertices = (prefix.reshape(-1).tolist())[1:-1]
        num = len(vertices)//3
        file = open(file_path, "w")
        for i in range(num):
            file.write(f"v {vertices[3*i+2]} {vertices[3*i+1]} {vertices[3*i]}\n")
        file.close()
        return

def make_padding_mask(input_ids, padding_idx=258):
    """True for pad tokens"""
    padding_mask = input_ids.eq(258)
    #print(input_ids)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _prepare_decoder_inputs(config,
                            input_ids,
                            decoder_input_ids,
                            causal_mask_dtype=torch.float32):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id

    bsz, tgt_len = decoder_input_ids.size()

    decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)

    # never mask leading token, even if it is pad
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]
    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=causal_mask_dtype,
                         device=decoder_input_ids.device)
    return decoder_input_ids, decoder_padding_mask, causal_mask


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
                                        self.head_dim).transpose(0, 1)  # [bs*heads, seq, head_dim]

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
    


    # def forward(
    #     self,
    #     query,
    #     key: Optional[Tensor],
    #     key_padding_mask: Optional[Tensor] = None,
    #     attn_mask: Optional[Tensor] = None,
    # ) -> Tuple[Tensor, Optional[Tensor]]:
    #     """
    #     Compute the attention output. You need to apply key_padding_mask and attn_mask before softmax operation.

    #     Args:
    #         query (torch.Tensor): The input query tensor, shape (seq_len, batch_size, embed_dim).
    #         key (Optional[torch.Tensor]): The input key tensor, shape (seq_len, batch_size, embed_dim).
    #                                      If None, it's assumed to be the same as the query tensor.
    #         key_padding_mask (Optional[torch.Tensor]): The key padding mask tensor, shape (batch_size, seq_len).
    #                                                   Default: None
    #         attn_mask (Optional[torch.Tensor]): The attention mask tensor, shape (seq_len, seq_len).
    #                                            Default: None

    #     Returns:
    #         attn_output (torch.Tensor): The attention output tensor, shape (seq_len, batch_size, embed_dim).

    #     """
    #     ##############################################################################
    #     #                  TODO: You need to complete the code here                  #
    #     ##############################################################################
    #     seqQ, bs, embed_dim = query.shape
    #     seqK = key.shape[0]
    #     #print(key.shape, query.shape)   # torch.Size([8, 1, 512]) torch.Size([1, 2, 512])
    #     Q = self.q_proj(query)
    #     K = self.k_proj(key)
    #     V = self.v_proj(key)
    #     Q = Q.reshape(seqQ, bs, self.num_heads, self.head_dim).permute(1,2,0,3).reshape(-1,seqQ,self.head_dim)
    #     #print(seqQ) # 1
    #     #print(K.shape)  # 8,1,512
    #     #print(seqK, bs, self.num_heads, self.head_dim)  # 8, 2, 8, 64
    #     K = K.reshape(seqK, bs, self.num_heads, self.head_dim).permute(1,2,0,3).reshape(-1,seqK,self.head_dim)
    #     V = V.reshape(seqK, bs, self.num_heads, self.head_dim).permute(1,2,0,3).reshape(-1,seqK,self.head_dim)  
    #     #  [bs*heads, seq, head_dim]
        
    #     scores = self.scaling * torch.matmul(Q.cuda(), K.transpose(-2,-1).cuda())    #[bs*heads, seq, seq]
    #     scores = scores.cuda()
    #     if key_padding_mask is not None:
    #         key_padding_mask = (key_padding_mask.reshape(bs, 1, 1, seqK).repeat(1, self.num_heads, seqQ, 1)).reshape(-1, seqQ, seqK)
    #         scores = scores.masked_fill((key_padding_mask != 0).cuda(), -1e9)
    #     if attn_mask is not None:
    #         scores = scores.masked_fill((attn_mask.reshape(1,seqQ,seqK).repeat(bs*self.num_heads,1,1)!=0).cuda(), -1e9)
    #     p_attn = scores.softmax(dim = -1)
    #     p_attn = nn.Dropout(p=self.dropout)(p_attn) # [bs*heads, seq, seq]
    #     x = torch.matmul(p_attn, V) #[bs*heads, seq, head_dim]
    #     attn_output = x.reshape(bs, self.num_heads, seqQ, self.head_dim).permute(2,0,1,3).reshape(seqQ, bs, -1)
    #     attn_output = self.out_proj(attn_output)
    #     ##############################################################################
    #     #                              END OF YOUR CODE                              #
    #     ##############################################################################
    #     return attn_output


    # @torch.no_grad()
    # def generate(self, prefix, file_path):
    #     # prefix: [int, int, ...]   known coordinates
    #     prefix = torch.tensor([257]+prefix, device='cuda').reshape(1, -1)
    #     while True:
    #         logits = self.logits(prefix)    # [bs=1, seq, 259]
    #         idx = torch.multinomial(logits[0,-1, :].softmax(-1), 1)
    #         prefix = torch.cat([prefix, torch.tensor([[idx]], device='cuda')], axis=1)
    #         if idx == 256:
    #             break
    #         # if prefix.shape[1] > 290:
    #         #     break
    #     vertices = (prefix.reshape(-1).tolist())[1:-1]
    #     num = len(vertices)//3
    #     file = open(file_path, "w")
    #     for i in range(num):
    #         file.write(f"v {vertices[3*i+2]} {vertices[3*i+1]} {vertices[3*i]}\n")
    #     file.close()
    #     return