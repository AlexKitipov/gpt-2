import torch
import torch.nn as nn
import json
import os
from dataclasses import asdict

from .config import GPT2Config


class GPT2Embeddings(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.n_positions, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)

        return self.dropout(word_emb + pos_emb)


def causal_mask(seq_len, device):
    return (
        torch.tril(torch.ones((seq_len, seq_len), device=device))
        .unsqueeze(0)
        .unsqueeze(0)
    )


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.size()

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(qkv.size(-1) // 3, dim=-1)

        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        mask = causal_mask(seq_len, hidden_states.device)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        return self.c_proj(context)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc_out(self.act(self.fc_in(x)))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.embeddings = GPT2Embeddings(config)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.ln_f(hidden_states)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        return self.lm_head(hidden_states)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(
            os.path.join(save_directory, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory):
        with open(
            os.path.join(load_directory, "config.json"), "r", encoding="utf-8"
        ) as f:
            config_data = json.load(f)
        config = GPT2Config(**config_data)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu"
        )
        model.load_state_dict(state_dict)
        return model
