import torch
from torch import nn


class FFN(nn.Sequential):
    def __init__(self, cfg):
        super().__init__(
            nn.Linear(cfg['embed_dim'], cfg['embed_dim'] * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(cfg['embed_dim'] * 4, cfg['embed_dim']),
            nn.Dropout(cfg['drop_rate'])
        )


class MHSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embed_dim = cfg['embed_dim']
        self.n_heads = cfg['n_heads']
        self.head_size = self.embed_dim // self.n_heads
        self.scale = self.head_size ** -0.5

        self.to_qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.to_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x, mask=None):
        B, N, C = x.shape

        # (batch, seq_length, emb_dim*3)
        qkv = self.to_qkv(x)
        # (batch, seq_length, emb_dim, 3)
        qkv = qkv.reshape(B, N, -1, 3)
        # (batch, seq_length, emb_dim)
        q, k, v = qkv.permute(3, 0, 1, 2)

        # (batch, seq_length, n_heads, head_size)
        q = q.reshape(B, N, self.n_heads, self.head_size)
        k = k.reshape(B, N, self.n_heads, self.head_size)
        v = v.reshape(B, N, self.n_heads, self.head_size)

        # (batch, n_heads, seq_length, head_size)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # (batch, n_heads, seq_length, seq_length)
        scores = (self.scale * q) @ k.transpose(-1, -2)
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), -torch.inf)

        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)

        # (batch, n_heads, seq_length, head_size)
        attn = weights @ v
        # (batch, seq_length, n_heads, head_size)
        attn = attn.permute(0, 2, 1, 3)
        # (batch, seq_length, emb_dim)
        attn = attn.reshape(B, N, self.embed_dim)

        return self.to_out(attn)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn_norm = nn.LayerNorm(cfg['embed_dim'])
        self.attn = MHSA(cfg)

        self.ffn_norm = nn.LayerNorm(cfg['embed_dim'])
        self.ffn = FFN(cfg)

        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1]), diagonal=0).to(x.device)
        x = self.dropout(self.attn(self.attn_norm(x), mask)) + x
        return self.dropout(self.ffn(self.ffn_norm(x))) + x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_embedding = nn.Embedding(cfg['vocab_size'], cfg['embed_dim'])
        self.pos_embedding = nn.Embedding(cfg['ctx_len'], cfg['embed_dim'])

        self.transformers = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.out_norm = nn.LayerNorm(cfg['embed_dim'])
        self.to_out = nn.Linear(cfg['embed_dim'], cfg['vocab_size'], bias=False)
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        batch, seq_len = x.shape

        # (batch_size, seq_len, embed_dim)
        x = self.tok_embedding(x)
        pos = self.pos_embedding(torch.arange(seq_len, dtype=torch.long, device=x.device))

        x = x + pos
        x = self.dropout(x)

        x = self.transformers(x)
        x = self.out_norm(x)
        x = self.to_out(x)

        return x
