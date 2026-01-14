import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                raise NotImplementedError
                # inputs = module(inputs)
        return inputs
    
class CrossAttention(nn.Module):
    """
    Simple Generic CrossAttention (based on LXMERT). 
    x: source query
    context: source of Key/Value
    key_padding_mask (True = ignore)
    Without causal masking.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, context, key_padding_mask=None):
        # x: Query [B, T_x, C]
        # context: Key/Value [B, T_ctx, C]
        # key_padding_mask: [B, T_ctx] bool
        
        B, T_x, C = x.size()
        B, T_ctx, _ = context.size()

        # compute querys, keys and values
        q = self.query(x).view(B, T_x, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(context).view(B, T_ctx, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(context).view(B, T_ctx, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention (B, nh, T_x, T_ctx)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if key_padding_mask is not None:
            # Expand mask for broadcasting: [B, 1, 1, T_ctx]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(mask, float("-inf"))
            fully_masked = mask.all(dim=-1, keepdim=True)
            att = torch.where(fully_masked, torch.zeros_like(att), att)
            att = F.softmax(att, dim=-1)
            att = att.masked_fill(mask, 0.0)
        else:
            att = F.softmax(att, dim=-1)
        
        att = self.attn_drop(att)
        
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T_x, C)
        
        y = self.resid_drop(self.proj(y))
        return y


class CoAttentionBlock(nn.Module):
    """
    Co-Attentional (ViLBERT style simplified).
    """
    def __init__(self, config):
        super().__init__()
        # Stream 1 (ex. States)
        self.ln1_s = nn.LayerNorm(config.n_embd)
        self.cross_attn_s = CrossAttention(config) # Q=S, K/V=A
        self.ln2_s = nn.LayerNorm(config.n_embd)
        self.mlp_s = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

        # Stream 2 (ex. Actions)
        self.ln1_a = nn.LayerNorm(config.n_embd)
        self.cross_attn_a = CrossAttention(config) # Q=A, K/V=S
        self.ln2_a = nn.LayerNorm(config.n_embd)
        self.mlp_a = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x_s, x_a, mask_s=None, mask_a=None):
        # Cross Attention (residual connection)
        x_s = x_s + self.cross_attn_s(self.ln1_s(x_s), self.ln1_a(x_a), key_padding_mask=mask_a)
        x_a = x_a + self.cross_attn_a(self.ln1_a(x_a), self.ln1_s(x_s), key_padding_mask=mask_s)

        # Feed Forward
        x_s = x_s + self.mlp_s(self.ln2_s(x_s))
        x_a = x_a + self.mlp_a(self.ln2_a(x_a))
        
        return x_s, x_a


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = config.n_head

    def forward(self, x, mask):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # slice + bool mask
        mask = mask[:, :, :T, :T]
        mask_bool = mask != 0

        att = att.masked_fill(~mask_bool, float("-inf"))
        # detect fully-masked query rows: [B, 1|nh, T, 1]
        fully_masked = ~mask_bool.any(dim=-1, keepdim=True)
        # replace fully-masked rows with zeros BEFORE softmax (prevents NaNs)
        att = torch.where(fully_masked, torch.zeros_like(att), att)

        # softmax + force masked positions to 0
        att = F.softmax(att, dim=-1)
        att = att * mask_bool.to(dtype=att.dtype)

        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TwinQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.n_embd
        self.block_1 = mySequential(*[Block(config) for _ in range(config.n_layer)])
        self.block_2 = mySequential(*[Block(config) for _ in range(config.n_layer)])
        self.q1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, x, mask):
        x1 = self.block_1(self.drop(x), mask)
        q_1 = self.q1(self.ln_f(x1))

        x2 = self.block_2(self.drop(x), mask)
        q_2 = self.q2(self.ln_f(x2))
        return q_1, q_2
