import torch
from torch import nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ClsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = (self.to_q(x[:, :1, :]), *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attention weight (softmax)
        cls_attn = self.attend(dots)
        # final attention
        out = torch.matmul(cls_attn, v)
        # concat each head attention
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MFM(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ClsAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                nn.Identity()
            ]))

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        for cls_attn, ff in self.layers:
            x = cls_attn(x) + x[:, :1, :]  # 8*1*768
            x = torch.cat((x, x2), dim=1)  # 8*9*768
        x = x[:, 0, :]  # 8*768
        return x

