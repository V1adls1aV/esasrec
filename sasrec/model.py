import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class CausalLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def feature_map(self, x):
        return F.elu(x) + 1.0

    def forward(self, query, key, value, **kwargs):
        L, B, E = query.shape

        Q = (
            self.q_proj(query)
            .view(L, B, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )
        K = (
            self.k_proj(key)
            .view(L, B, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )
        V = (
            self.v_proj(value)
            .view(L, B, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )

        Q = self.feature_map(Q)
        K = self.feature_map(K)

        KV = torch.einsum("bhld,bhlm->bhldm", K, V)
        KV_cumsum = torch.cumsum(KV, dim=2)

        K_cumsum = torch.cumsum(K, dim=2)

        num = torch.einsum("bhld,bhldm->bhlm", Q, KV_cumsum)
        den = torch.einsum("bhld,bhld->bhl", Q, K_cumsum)

        out = num / (den.unsqueeze(-1) + 1e-6)

        out = out.permute(2, 0, 1, 3).contiguous().view(L, B, E)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out, None


class MambaLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)

    def forward(self, query, key, value, attn_mask=None, **kwargs):
        x = query.transpose(0, 1)
        out = self.mamba(x)
        out = out.transpose(0, 1)
        return out, None


class SASRec(nn.Module):
    def __init__(
        self,
        item_num,
        maxlen=800,
        hidden_units=256,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.1,
        initializer_range=0.02,
        attn_types=None,
        layers_mask=None,
    ):
        super().__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range

        if attn_types is None:
            self.attn_types = ["standard"] * num_blocks
        else:
            assert len(attn_types) == num_blocks, (
                "Длина списка attn_types должна совпадать с num_blocks"
            )
            self.attn_types = attn_types
        if layers_mask is None:
            self.layers_mask = [1] * num_blocks
        else:
            self.layers_mask = [int(i) for i in layers_mask]

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for i in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))

            if self.attn_types[i] in ("linear", "l"):
                self.attention_layers.append(
                    CausalLinearAttention(hidden_units, num_heads, dropout_rate)
                )
            elif self.attn_types[i] in ("mamba", "m", "mamba_noff", "mnff"):
                self.attention_layers.append(MambaLayer(hidden_units))
            elif self.attn_types[i] in ("standard", "s"):
                self.attention_layers.append(
                    nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
                )
            else:
                raise ValueError(
                    f"There is not attention of type <{self.attn_types[i]}>"
                )

            if self.attn_types[i] in ("mamba_noff", "mnff"):
                self.forward_layernorms.append(nn.Identity())
                self.forward_layers.append(nn.Identity())
            else:
                self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
                self.forward_layers.append(
                    PointWiseFeedForward(hidden_units, dropout_rate)
                )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids):
        seqs = self.item_emb(input_ids)
        seqs *= self.hidden_units**0.5

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
            0
        )
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = input_ids == 0
        seqs = seqs * (~timeline_mask).unsqueeze(-1).float()

        tl = seqs.shape[1]
        attn_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=seqs.device)
        )

        for i in range(self.num_blocks):
            if self.layers_mask[i] == 0:
                continue
            seqs_t = seqs.transpose(0, 1)
            Q = self.attention_layernorms[i](seqs_t)

            mha_out, _ = self.attention_layers[i](
                Q, seqs_t, seqs_t, attn_mask=attn_mask
            )

            seqs_t = Q + mha_out
            seqs = seqs_t.transpose(0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

            seqs = seqs * (~timeline_mask).unsqueeze(-1).float()

        return self.last_layernorm(seqs)
