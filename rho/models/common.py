import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A simple MLP with optional hidden layer.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class BasicCNN(nn.Module):
    def __init__(self, input_channels, num_actions, conv_channels=None, fc_hidden=256):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        self.conv1 = nn.Conv2d(input_channels, conv_channels[0], kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2)
        self.fc1 = nn.Linear(conv_channels[2] * 10 * 10, fc_hidden)  # Adjust 10*10 based on input image size
        self.fc2 = nn.Linear(fc_hidden, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GELUTanh(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), GELUTanh(), nn.Linear(ff_dim, embed_dim)
        )
        self.pre_norm = GemmaRMSNorm(embed_dim)
        self.post_norm = GemmaRMSNorm(embed_dim)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_output

        x_norm = self.post_norm(x)
        ff_out = self.feedforward(x_norm)
        x = x + ff_out
        return x


class TransformerBlock2(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), GELUTanh(), nn.Linear(ff_dim, embed_dim)
        )
        self.pre_norm = GemmaRMSNorm(embed_dim)
        self.post_norm = GemmaRMSNorm(embed_dim)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        # qkv = self.qkv_proj(x_norm)  # This line should exist to compute qkv
        # q, k, v = qkv.chunk(3, dim=-1)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_output

        x_norm = self.post_norm(x)
        ff_out = self.feedforward(x_norm)
        x = x + ff_out
        return x
