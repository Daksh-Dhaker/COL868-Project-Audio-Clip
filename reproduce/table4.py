#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import os
import shutil
import tempfile
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import ftfy
import librosa
import numpy as np
import pandas as pd
import regex as re
import scipy.signal as sps
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


# =========================
# Safe loading / utilities
# =========================

def _safe_torch_load(path: str):
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def looks_like_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        head = path.read_bytes()[:120]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def resolve_checkpoint_path(path: Path) -> Path:
    release_map = {
        "AudioCLIP-Full-Training.pt": "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt",
        "AudioCLIP-Partial-Training.pt": "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt",
    }
    if not looks_like_lfs_pointer(path):
        return path

    url = release_map.get(path.name)
    if url is None:
        raise RuntimeError(
            f"Checkpoint is a Git LFS pointer and cannot be auto-downloaded: {path}. Provide a real .pt file."
        )

    out_dir = Path("/kaggle/working") / "models" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    if out_path.exists() and not looks_like_lfs_pointer(out_path):
        print(f"Using downloaded checkpoint: {out_path}")
        return out_path

    print(f"Detected LFS pointer for {path.name}. Downloading real checkpoint -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


# =========================
# CLIP tokenizer (vendored)
# =========================

BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


def _is_valid_gzip(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with gzip.open(path, "rb") as handle:
            handle.read(2)
        return True
    except OSError:
        return False


def _download_bpe(target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="audioclip_bpe_", suffix=".gz")
    os.close(fd)
    try:
        urllib.request.urlretrieve(BPE_URL, tmp_path)
        if not _is_valid_gzip(tmp_path):
            raise RuntimeError(f"Downloaded BPE file is invalid: {tmp_path}")
        shutil.move(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return target_path


def _resolve_bpe_path() -> str:
    env_path = os.environ.get("AUDIOCLIP_BPE_PATH")
    if env_path:
        if looks_like_lfs_pointer(Path(env_path)):
            raise RuntimeError(
                f"AUDIOCLIP_BPE_PATH points to a Git LFS pointer file: {env_path}. "
                "Please provide a real bpe_simple_vocab_16e6.txt.gz file."
            )
        if not _is_valid_gzip(env_path):
            raise RuntimeError(
                f"AUDIOCLIP_BPE_PATH is not a valid gzip file: {env_path}. "
                "Please provide a real bpe_simple_vocab_16e6.txt.gz file."
            )
        return env_path

    repo_bpe = Path(__file__).resolve().parent.parent / "assets" / BPE_FILENAME
    if _is_valid_gzip(str(repo_bpe)) and not looks_like_lfs_pointer(repo_bpe):
        return str(repo_bpe)

    cache_bpe = os.path.join(os.path.expanduser("~"), ".cache", "audioclip", BPE_FILENAME)
    if _is_valid_gzip(cache_bpe):
        return cache_bpe

    return _download_bpe(cache_bpe)


@lru_cache()
def default_bpe():
    return _resolve_bpe_path()


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


_TOKENIZER = SimpleTokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _TOKENIZER.encoder["<|startoftext|>"]
    eot_token = _TOKENIZER.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _TOKENIZER.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


# =========================
# CLIP model (vendored)
# =========================

class CLIPBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride > 1 or inplanes != planes * CLIPBottleneck.expansion:
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x[0]


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [CLIPBottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * CLIPBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(CLIPBottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x_):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x_ = self.relu(bn(conv(x_)))
            x_ = self.avgpool(x_)
            return x_

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


# =========================
# ESResNet/FBSP (vendored)
# =========================

def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((old_value - old_min) * new_range) / old_range) + new_min


def frame_signal(signal: torch.Tensor, frame_length: int, hop_length: int, window: torch.Tensor = None) -> torch.Tensor:
    if window is None:
        window = torch.ones(frame_length, dtype=signal.dtype, device=signal.device)

    if window.shape[0] != frame_length:
        raise ValueError(f"Wrong window length: expected {frame_length}, got {window.shape[0]}")

    signal_length = signal.shape[-1]
    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((1.0 * signal_length - frame_length) / hop_length))

    pad_len = int((num_frames - 1) * hop_length + frame_length)
    if pad_len > signal_length:
        zeros = torch.zeros(pad_len - signal_length, device=signal.device, dtype=signal.dtype)
        while zeros.dim() < signal.dim():
            zeros.unsqueeze_(0)
        pad_signal = torch.cat((zeros.expand(*signal.shape[:-1], -1)[..., : zeros.shape[-1] // 2], signal), dim=-1)
        pad_signal = torch.cat((pad_signal, zeros.expand(*signal.shape[:-1], -1)[..., zeros.shape[-1] // 2 :]), dim=-1)
    else:
        pad_signal = signal

    indices = torch.arange(0, frame_length, device=signal.device).repeat(num_frames, 1)
    indices += (
        torch.arange(0, num_frames * hop_length, hop_length, device=signal.device)
        .repeat(frame_length, 1)
        .t_()
    )
    indices = indices.long()

    frames = pad_signal[..., indices]
    frames = frames * window
    return frames


class Attention2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int,
        kernel_size: Tuple[int, int],
        padding_size: Tuple[int, int],
    ):
        super().__init__()
        self.conv_depth = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * num_kernels,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=in_channels,
        )
        self.conv_point = torch.nn.Conv2d(
            in_channels=in_channels * num_kernels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = F.adaptive_max_pool2d(x, size)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride=1, groups: int = 1, dilation: Union[int, Tuple[int, int]] = 1):
    return torch.nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: Union[int, Tuple[int, int]] = 1):
    return torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False)


class ESBasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Union[int, Tuple[int, int]] = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: Union[int, Tuple[int, int]] = 1,
        norm_layer: Optional[Type[torch.nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ESBottleneck(torch.nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Union[int, Tuple[int, int]] = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: Union[int, Tuple[int, int]] = 1,
        norm_layer: Optional[Type[torch.nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = torch.nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetWithAttention(torch.nn.Module):
    def __init__(
        self,
        block: Type[Union[ESBasicBlock, ESBottleneck]],
        layers: List[int],
        apply_attention: bool = False,
        num_channels: int = 3,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: bool = None,
        norm_layer: Optional[Type[torch.nn.Module]] = None,
    ):
        super().__init__()
        self.apply_attention = apply_attention

        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = torch.nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if self.apply_attention:
            self.att1 = Attention2d(64, 64 * block.expansion, 1, (3, 1), (1, 0))

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        if self.apply_attention:
            self.att2 = Attention2d(64 * block.expansion, 128 * block.expansion, 1, (1, 5), (0, 2))

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        if self.apply_attention:
            self.att3 = Attention2d(128 * block.expansion, 256 * block.expansion, 1, (3, 1), (1, 0))

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if self.apply_attention:
            self.att4 = Attention2d(256 * block.expansion, 512 * block.expansion, 1, (1, 5), (0, 2))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        if self.apply_attention:
            self.att5 = Attention2d(512 * block.expansion, 512 * block.expansion, 1, (3, 5), (1, 2))

        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ESBottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ESBasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[ESBasicBlock, ESBottleneck]],
        planes: int,
        blocks: int,
        stride: Union[int, Tuple[int, int]] = 1,
        dilate: bool = False,
    ) -> torch.nn.Module:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return torch.nn.Sequential(*layers)

    def _forward_pre_processing(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.get_default_dtype())

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.apply_attention:
            x_att = x.clone()
            x = self.layer1(x)
            x_att = self.att1(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer2(x)
            x_att = self.att2(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer3(x)
            x_att = self.att3(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer4(x)
            x_att = self.att4(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x

    def _forward_reduction(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_attention:
            x_att = x.clone()
            x = self.avgpool(x)
            x_att = self.att5(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.avgpool(x)

        x = torch.flatten(x, 1)
        return x

    def _forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = self._forward_pre_processing(x)
        x = self._forward_features(x)
        x = self._forward_reduction(x)
        y_pred = self._forward_classifier(x)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).mean()

        return y_pred if loss is None else (y_pred, loss)

    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(y_pred, tuple):
            y_pred, *_ = y_pred

        if y_pred.shape == y.shape:
            return F.binary_cross_entropy_with_logits(
                y_pred,
                y.to(dtype=y_pred.dtype, device=y_pred.device),
                reduction="sum",
            ) / y_pred.shape[0]
        return F.cross_entropy(y_pred, y.to(y_pred.device))


class _ESResNet(ResNetWithAttention):
    loading_func = staticmethod(tv.models.resnet50)

    def __init__(
        self,
        block: Type[Union[ESBasicBlock, ESBottleneck]],
        layers: List[int],
        apply_attention: bool = False,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[str] = None,
        normalized: bool = False,
        onesided: bool = True,
        spec_height: int = 224,
        spec_width: int = 224,
        num_classes: int = 1000,
        pretrained: Union[bool, str] = False,
        lock_pretrained: Optional[Union[bool, List[str]]] = None,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: bool = None,
        norm_layer: Optional[Type[torch.nn.Module]] = None,
    ):
        super().__init__(
            block=block,
            layers=layers,
            apply_attention=apply_attention,
            num_channels=3,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

        self.num_classes = num_classes
        self.fc = torch.nn.Linear(self.fc.in_features, self.num_classes, bias=self.fc.bias is not None)

        if hop_length is None:
            hop_length = int(np.floor(n_fft / 4))
        if win_length is None:
            win_length = n_fft
        if window is None:
            window = "boxcar"

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalized = normalized
        self.onesided = onesided
        self.spec_height = spec_height
        self.spec_width = spec_width

        self.pretrained = pretrained
        self._inject_members()

        if pretrained:
            err_msg = self.load_pretrained()
            unlocked_weights = []
            for name, p in self.named_parameters():
                unlock = True
                if isinstance(lock_pretrained, bool):
                    if lock_pretrained and name not in err_msg:
                        unlock = False
                elif isinstance(lock_pretrained, list):
                    if name in lock_pretrained:
                        unlock = False
                p.requires_grad_(unlock)
                if unlock:
                    unlocked_weights.append(name)
            print(f"Following weights are unlocked: {unlocked_weights}")

        window_buffer: torch.Tensor = torch.from_numpy(sps.get_window(window=window, Nx=win_length, fftbins=True)).to(torch.get_default_dtype())
        self.register_buffer("window", window_buffer)
        self.log10_eps = 1e-18

    def _inject_members(self):
        pass

    def load_pretrained(self) -> str:
        if isinstance(self.pretrained, bool):
            state_dict = self.loading_func(pretrained=True).state_dict()
        else:
            state_dict = _safe_torch_load(self.pretrained)

        err_msg = ""
        try:
            self.load_state_dict(state_dict=state_dict, strict=True)
        except RuntimeError as ex:
            err_msg += f"While loading some errors occurred.\n{ex}"
            print(err_msg)
        return err_msg

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x.view(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            pad_mode="reflect",
            normalized=self.normalized,
            onesided=True,
            return_complex=False,
        )

        if not self.onesided:
            spec = torch.cat((torch.flip(spec, dims=(-3,)), spec), dim=-3)

        return spec

    def split_spectrogram(self, spec: torch.Tensor, batch_size: int) -> torch.Tensor:
        spec_height_per_band = spec.shape[-3] // self.conv1.in_channels
        spec_height_single_band = self.conv1.in_channels * spec_height_per_band
        spec = spec[:, :spec_height_single_band]
        spec = spec.reshape(batch_size, -1, spec.shape[-3] // self.conv1.in_channels, *spec.shape[-2:])
        return spec

    def spectrogram_to_power(self, spec: torch.Tensor) -> torch.Tensor:
        spec_height = spec.shape[-3] if self.spec_height < 1 else self.spec_height
        spec_width = spec.shape[-2] if self.spec_width < 1 else self.spec_width

        pow_spec = spec[..., 0] ** 2 + spec[..., 1] ** 2

        if spec_height != pow_spec.shape[-2] or spec_width != pow_spec.shape[-1]:
            pow_spec = F.interpolate(pow_spec, size=(spec_height, spec_width), mode="bilinear", align_corners=True)

        return pow_spec

    def _forward_pre_processing(self, x: torch.Tensor) -> torch.Tensor:
        x = super()._forward_pre_processing(x)
        x = scale(x, -32768.0, 32767, -1.0, 1.0)

        spec = self.spectrogram(x)
        spec_split_ch = self.split_spectrogram(spec, x.shape[0])
        pow_spec_split_ch = self.spectrogram_to_power(spec_split_ch)
        pow_spec_split_ch = torch.where(
            cast(torch.Tensor, pow_spec_split_ch > 0.0),
            pow_spec_split_ch,
            torch.full_like(pow_spec_split_ch, self.log10_eps),
        )
        pow_spec_split_ch = pow_spec_split_ch.reshape(x.shape[0], -1, self.conv1.in_channels, *pow_spec_split_ch.shape[-2:])
        x_db = torch.log10(pow_spec_split_ch).mul(10.0)

        return x_db

    def _forward_features(self, x_db: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for ch_idx in range(x_db.shape[1]):
            ch = x_db[:, ch_idx]
            out = super()._forward_features(ch)
            outputs.append(out)
        return outputs

    def _forward_reduction(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for ch in x:
            out = super()._forward_reduction(ch)
            outputs.append(out)
        return torch.stack(outputs, dim=-1).sum(dim=-1)


class LinearFBSP(torch.nn.Module):
    def __init__(self, out_features: int, bias: bool = True, normalized: bool = False):
        super().__init__()
        self.out_features = out_features
        self.normalized = normalized
        self.eps = 1e-8

        default_dtype = torch.get_default_dtype()
        self.register_parameter("m", torch.nn.Parameter(torch.zeros(self.out_features, dtype=default_dtype)))
        self.register_parameter("fb", torch.nn.Parameter(torch.ones(self.out_features, dtype=default_dtype)))
        self.register_parameter("fc", torch.nn.Parameter(torch.arange(self.out_features, dtype=default_dtype)))
        self.register_parameter(
            "bias",
            torch.nn.Parameter(torch.normal(0.0, 0.5, (self.out_features, 2), dtype=default_dtype)
                               if bias
                               else cast(torch.nn.Parameter, None)),
        )

        self.m.register_hook(lambda grad: grad / (torch.norm(grad, p=float("inf")) + self.eps))
        self.fb.register_hook(lambda grad: grad / (torch.norm(grad, p=float("inf")) + self.eps))
        self.fc.register_hook(lambda grad: grad / (torch.norm(grad, p=float("inf")) + self.eps))

    @staticmethod
    def power(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        magnitudes = (x1[..., 0] ** 2 + x1[..., 1] ** 2) ** 0.5
        phases = x1[..., 1].atan2(x1[..., 0])

        power_real = x2[..., 0]
        power_imag = x2[..., 1]

        mag_out = ((magnitudes ** 2) ** (0.5 * power_real) * torch.exp(-power_imag * phases))

        return mag_out.unsqueeze(-1) * torch.stack(
            (
                (power_real * phases + 0.5 * power_imag * (magnitudes ** 2).log()).cos(),
                (power_real * phases + 0.5 * power_imag * (magnitudes ** 2).log()).sin(),
            ),
            dim=-1,
        )

    @staticmethod
    def sinc(x: torch.Tensor) -> torch.Tensor:
        return torch.where(cast(torch.Tensor, x == 0), torch.ones_like(x), torch.sin(x) / x)

    def _materialize_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        x_is_complex = x.shape[-1] == 2
        in_features = x.shape[-1 - int(x_is_complex)]

        t = np.pi * torch.linspace(-1.0, 1.0, in_features, dtype=x.dtype, device=x.device).reshape(1, -1, 1) + self.eps

        m = self.m.reshape(-1, 1, 1)
        fb = self.fb.reshape(-1, 1, 1)
        fc = self.fc.reshape(-1, 1, 1)

        kernel = torch.cat((torch.cos(fc * t), -torch.sin(fc * t)), dim=-1)
        scale = fb.sqrt()
        win = self.sinc(fb * t / (m + self.eps))
        win = self.power(torch.cat((win, torch.zeros_like(win)), dim=-1), torch.cat((m, torch.zeros_like(m)), dim=-1))

        weights = scale * torch.cat(
            (
                win[..., :1] * kernel[..., :1] - win[..., 1:] * kernel[..., 1:],
                win[..., :1] * kernel[..., 1:] + win[..., 1:] * kernel[..., :1],
            ),
            dim=-1,
        )

        if self.normalized:
            weights = weights / (in_features ** 0.5)

        return weights, x_is_complex

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, x_is_complex = self._materialize_weights(x)

        if x_is_complex:
            x = torch.stack(
                (
                    F.linear(x[..., 0], weights[..., 0]) - F.linear(x[..., 1], weights[..., 1]),
                    F.linear(x[..., 0], weights[..., 1]) + F.linear(x[..., 1], weights[..., 0]),
                ),
                dim=-1,
            )
        else:
            x = torch.stack((F.linear(x, weights[..., 0]), F.linear(x, weights[..., 1])), dim=-1)

        if (self.bias is not None) and (self.bias.numel() == (self.out_features * 2)):
            x = x + self.bias

        return x, weights


ttf_weights = {}


class _ESResNetFBSP(_ESResNet):
    def _inject_members(self):
        self.add_module(
            "fbsp",
            LinearFBSP(
                out_features=int(round(self.n_fft / 2)) + 1 if self.onesided else self.n_fft,
                normalized=self.normalized,
                bias=False,
            ),
        )

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            frames = frame_signal(
                signal=x.view(-1, x.shape[-1]),
                frame_length=self.win_length,
                hop_length=self.hop_length,
                window=self.window,
            )

            if self.n_fft > self.win_length:
                pad_length = self.n_fft - self.win_length
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                frames = F.pad(frames, [pad_left, pad_right])

        spec, ttf_weights_ = self.fbsp(frames)
        spec = spec.transpose(-2, -3)
        ttf_weights[x.device] = ttf_weights_
        return spec

    def loss_ttf(self, device: torch.device) -> torch.Tensor:
        ttf_norm = torch.norm(ttf_weights[device], p=2, dim=[-1, -2])
        return F.mse_loss(
            ttf_norm,
            torch.full_like(ttf_norm, 1.0 if self.normalized else self.n_fft ** 0.5),
        )


class ESResNeXtFBSP(_ESResNetFBSP):
    loading_func = staticmethod(tv.models.resnext50_32x4d)

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[str] = None,
        normalized: bool = False,
        onesided: bool = True,
        spec_height: int = 224,
        spec_width: int = 224,
        num_classes: int = 1000,
        apply_attention: bool = False,
        pretrained: Union[bool, str] = False,
        lock_pretrained: Optional[Union[bool, List[str]]] = None,
    ):
        super().__init__(
            block=ESBottleneck,
            layers=[3, 4, 6, 3],
            apply_attention=apply_attention,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=num_classes,
            pretrained=pretrained,
            lock_pretrained=lock_pretrained,
            groups=32,
            width_per_group=4,
        )


# =========================
# AudioCLIP (vendored)
# =========================

ClipFeatures = Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
ClipLogits = Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
ClipOutput = Tuple[Tuple[ClipFeatures, ClipLogits], Optional[torch.Tensor]]


class AudioCLIP(CLIP):
    def __init__(
        self,
        embed_dim: int = 1024,
        image_resolution: int = 224,
        vision_layers: Union[Tuple[int, int, int, int], int] = (3, 4, 6, 3),
        vision_width: int = 64,
        vision_patch_size: Optional[int] = None,
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        n_fft: int = 2048,
        hop_length: Optional[int] = 561,
        win_length: Optional[int] = 1654,
        window: Optional[str] = "blackmanharris",
        normalized: bool = True,
        onesided: bool = True,
        spec_height: int = -1,
        spec_width: int = -1,
        apply_attention: bool = True,
        multilabel: bool = True,
        pretrained: Union[bool, str] = True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

        self.audio = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=False,
        )

        self.multilabel = multilabel
        self.pretrained = pretrained

        self.logit_scale_ai = torch.nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_at = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

        if isinstance(self.pretrained, str):
            self.load_state_dict(_safe_torch_load(self.pretrained), strict=False)
        elif self.pretrained:
            assets = Path(__file__).resolve().parent.parent / "assets"
            self.load_state_dict(_safe_torch_load(str(assets / "CLIP.pt")), strict=False)
            try:
                self.audio.load_state_dict(_safe_torch_load(str(assets / "ESRNXFBSP.pt")), strict=False)
            except RuntimeError as ex:
                print(ex)
                print("Audio weights loaded")

        self.embed_dim = embed_dim

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio(audio.to(self.device))

    def encode_text(
        self,
        text: List[List[str]],
        base_str: str = "{}",
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch_indices is not None:
            text = [text[idx] for idx in batch_indices]

        text_joined = [", ".join(entities) for entities in text]
        text_tokens = torch.cat([tokenize(base_str.format(entities)) for entities in text_joined])
        text_tokens = text_tokens.to(self.device)
        return super().encode_text(text_tokens)

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[List[List[str]]] = None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> ClipOutput:
        audio_features = None
        image_features = None
        text_features = None

        if audio is not None:
            audio_features = self.encode_audio(audio)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        if image is not None:
            image_features = self.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if text is not None:
            if batch_indices is None:
                batch_indices = torch.arange(len(text), dtype=torch.int64, device=self.device)
            text_features = self.encode_text(text, "{}", batch_indices)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        features: ClipFeatures = (audio_features, image_features, text_features)

        logit_scale_ai = torch.clamp(self.logit_scale_ai.exp(), min=1.0, max=100.0)
        logit_scale_at = torch.clamp(self.logit_scale_at.exp(), min=1.0, max=100.0)
        logit_scale_it = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)

        logits_audio_image = None
        logits_audio_text = None
        logits_image_text = None

        if (audio_features is not None) and (image_features is not None):
            logits_audio_image = logit_scale_ai * audio_features @ image_features.T

        if (audio_features is not None) and (text_features is not None):
            logits_audio_text = logit_scale_at * audio_features @ text_features.T

        if (image_features is not None) and (text_features is not None):
            logits_image_text = logit_scale_it * image_features @ text_features.T

        logits: ClipLogits = (logits_audio_image, logits_audio_text, logits_image_text)
        return (features, logits), None


# =========================
# Retrieval evaluation
# =========================

@dataclass
class Item:
    key: str
    labels: set[str]
    fold: int | None = None
    path: Path | None = None
    text: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Single-file Table-4 retrieval evaluator for AudioCLIP "
            "with vendored model definitions (no repo imports)."
        )
    )
    p.add_argument("--dataset", required=True, choices=["esc50", "us8k", "imagenet", "audioset"])
    p.add_argument("--query-type", required=True, choices=["text", "audio", "image"])
    p.add_argument("--result-type", required=True, choices=["text", "audio", "image"])
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audio-length", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--protocol",
        choices=["paper", "global"],
        default="paper",
        help="paper: fold-wise aggregation for esc50/us8k; global: evaluate all items together.",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional specific fold (esc50:1-5, us8k:1-10). If omitted, all folds are averaged in paper mode.",
    )
    p.add_argument("--prompt-template", default="{}", help="Template used for text queries, e.g. 'a sound of {}'.")
    return p.parse_args()


def default_audio_length(dataset: str) -> int:
    return 176_400 if dataset == "us8k" else 220_500


def image_transform() -> Compose:
    return Compose(
        [
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda im: im.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


class ToTensor1D:
    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t


class CenterPadCrop1D:
    def __init__(self, out_len: int):
        self.out_len = out_len

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        if t < self.out_len:
            left = int(round(0.5 * (self.out_len - t)))
            right = self.out_len - (left + t)
            pad_left_val = x[..., 0:1].mean().to(x.dtype)
            pad_right_val = x[..., -1:].mean().to(x.dtype)
            left_pad = torch.full(x.shape[:-1] + (left,), pad_left_val.item(), dtype=x.dtype, device=x.device)
            right_pad = torch.full(x.shape[:-1] + (right,), pad_right_val.item(), dtype=x.dtype, device=x.device)
            x = torch.cat((left_pad, x, right_pad), dim=-1)
        elif t > self.out_len:
            left = int(round(0.5 * (t - self.out_len)))
            x = x[..., left : left + self.out_len]
        return x


def audio_transform(out_len: int) -> Compose:
    return Compose([ToTensor1D(), CenterPadCrop1D(out_len=out_len)])


def normalize_labels(labels: Iterable[str]) -> set[str]:
    return {str(lb).replace("_", " ").strip() for lb in labels if str(lb).strip()}


def load_esc50(root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    meta_path = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    if not meta_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {meta_path}")

    meta = pd.read_csv(meta_path)
    audio_items: list[Item] = []
    for _, row in meta.iterrows():
        label = str(row["category"]).replace("_", " ")
        path = audio_dir / str(row["filename"])
        if path.exists():
            audio_items.append(Item(key=path.stem, labels={label}, fold=int(row["fold"]), path=path))

    labels = sorted({next(iter(it.labels)) for it in audio_items})
    text_items = [Item(key=lb, labels={lb}, text=lb) for lb in labels]
    return text_items, audio_items, []


def load_us8k(root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    meta_path = root / "metadata" / "UrbanSound8K.csv"
    audio_root = root / "audio"
    if not meta_path.exists():
        flat_meta = root / "UrbanSound8K.csv"
        if flat_meta.exists():
            meta_path = flat_meta
            audio_root = root
    if not meta_path.exists():
        raise FileNotFoundError(f"UrbanSound8K metadata not found: {meta_path}")

    meta = pd.read_csv(meta_path)
    audio_items: list[Item] = []
    for _, row in meta.iterrows():
        label = str(row["class"]).replace("_", " ").strip()
        fold = int(row["fold"])
        path = audio_root / f"fold{fold}" / str(row["slice_file_name"])
        if path.exists():
            audio_items.append(Item(key=path.stem, labels={label}, fold=fold, path=path))

    labels = sorted({next(iter(it.labels)) for it in audio_items})
    text_items = [Item(key=lb, labels={lb}, text=lb) for lb in labels]
    return text_items, audio_items, []


def load_jsonl_manifest(root: Path, name: str) -> list[Item]:
    manifest = root / f"{name}.jsonl"
    if not manifest.exists():
        return []

    items: list[Item] = []
    with manifest.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels = normalize_labels(row.get("labels", []))
            if not labels:
                continue

            key = str(row.get("key") or row.get("id") or f"{name}_{idx}")
            it = Item(key=key, labels=labels)
            if name in {"audio", "image"}:
                rel = Path(str(row.get("path", "")))
                if not rel:
                    continue
                it.path = rel if rel.is_absolute() else (root / rel)
            else:
                it.text = str(row.get("text") or next(iter(labels)))
            items.append(it)
    return items


def load_custom(root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    audio_items = load_jsonl_manifest(root, "audio")
    image_items = load_jsonl_manifest(root, "image")
    text_items = load_jsonl_manifest(root, "text")
    if not text_items:
        label_space = sorted({lb for it in audio_items + image_items for lb in it.labels})
        text_items = [Item(key=lb, labels={lb}, text=lb) for lb in label_space]
    return text_items, audio_items, image_items


def build_items(dataset: str, root: Path) -> tuple[list[Item], list[Item], list[Item]]:
    if dataset == "esc50":
        return load_esc50(root)
    if dataset == "us8k":
        return load_us8k(root)
    return load_custom(root)


def load_audio(path: Path, sr: int, tfm: Compose) -> torch.Tensor:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    wav = (wav.astype(np.float32) * 32768.0)[np.newaxis, :]
    return tfm(wav)


def load_image(path: Path, tfm: Compose) -> torch.Tensor:
    with Image.open(path) as im:
        return tfm(im)


def encode_text(model, items: list[Item], prompt_template: str, bs: int) -> torch.Tensor:
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            batch = items[i : i + bs]
            text = [[prompt_template.format(it.text or next(iter(it.labels)))] for it in batch]
            ((_, _, tf), _), _ = model(text=text)
            tf = tf / torch.linalg.norm(tf, dim=-1, keepdim=True)
            out.append(tf.detach().cpu())
    return torch.cat(out, dim=0)


def encode_audio(model, items: list[Item], bs: int, device: torch.device, sr: int, tfm: Compose) -> torch.Tensor:
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            b = items[i : i + bs]
            audio = torch.stack([load_audio(it.path, sr, tfm) for it in b]).to(device)
            ((af, _, _), _), _ = model(audio=audio)
            af = af / torch.linalg.norm(af, dim=-1, keepdim=True)
            out.append(af.detach().cpu())
    return torch.cat(out, dim=0)


def encode_image(model, items: list[Item], bs: int, device: torch.device, tfm: Compose) -> torch.Tensor:
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            b = items[i : i + bs]
            imgs = torch.stack([load_image(it.path, tfm) for it in b]).to(device)
            ((_, imf, _), _), _ = model(image=imgs)
            imf = imf / torch.linalg.norm(imf, dim=-1, keepdim=True)
            out.append(imf.detach().cpu())
    return torch.cat(out, dim=0)


def pair_logits(model, q_type: str, r_type: str, qf: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    pair = {q_type, r_type}
    if pair == {"audio", "image"}:
        s = torch.clamp(model.logit_scale_ai.exp().detach().cpu(), min=1.0, max=100.0)
    elif pair == {"audio", "text"}:
        s = torch.clamp(model.logit_scale_at.exp().detach().cpu(), min=1.0, max=100.0)
    elif pair == {"image", "text"}:
        s = torch.clamp(model.logit_scale.exp().detach().cpu(), min=1.0, max=100.0)
    else:
        s = torch.tensor(1.0)
    return s * (qf @ rf.T)


def relevance(queries: list[Item], results: list[Item]) -> np.ndarray:
    rel = np.zeros((len(queries), len(results)), dtype=np.bool_)
    for qi, q in enumerate(queries):
        for ri, r in enumerate(results):
            rel[qi, ri] = len(q.labels & r.labels) > 0
    return rel


def p_at_1(scores: np.ndarray, rel: np.ndarray) -> float:
    idx = np.argmax(scores, axis=1)
    return float(np.mean(rel[np.arange(rel.shape[0]), idx].astype(np.float32)))


def r_at_1(scores: np.ndarray, rel: np.ndarray) -> float:
    inv_idx = np.argmax(scores.T, axis=1)
    return float(np.mean(rel[inv_idx, np.arange(rel.shape[1])].astype(np.float32)))


def m_ap(scores: np.ndarray, rel: np.ndarray) -> float:
    aps: list[float] = []
    for i in range(scores.shape[0]):
        gt = rel[i]
        n_rel = int(gt.sum())
        if n_rel == 0:
            continue
        order = np.argsort(-scores[i])
        r = gt[order].astype(np.float32)
        c = np.cumsum(r)
        p = c / (np.arange(r.shape[0], dtype=np.float32) + 1.0)
        aps.append(float((p * r).sum() / n_rel))
    return float("nan") if not aps else float(np.mean(aps))


def evaluate_once(
    model,
    q_type: str,
    r_type: str,
    all_items: dict[str, list[Item]],
    batch_size: int,
    device: torch.device,
    sample_rate: int,
    audio_len: int,
    prompt_template: str,
) -> tuple[float, float, float, int, int]:
    q_items = all_items[q_type]
    r_items = all_items[r_type]
    if not q_items or not r_items:
        raise ValueError(f"No items for setting {q_type}->{r_type}")

    img_tf = image_transform()
    aud_tf = audio_transform(audio_len)

    feats: dict[str, torch.Tensor] = {}
    if "text" in {q_type, r_type}:
        feats["text"] = encode_text(model, all_items["text"], prompt_template, batch_size)
    if "audio" in {q_type, r_type}:
        feats["audio"] = encode_audio(model, all_items["audio"], batch_size, device, sample_rate, aud_tf)
    if "image" in {q_type, r_type}:
        feats["image"] = encode_image(model, all_items["image"], batch_size, device, img_tf)

    def select(mod: str, subset: list[Item]) -> torch.Tensor:
        key_to_idx = {it.key: i for i, it in enumerate(all_items[mod])}
        ids = [key_to_idx[it.key] for it in subset]
        return feats[mod][ids]

    qf = select(q_type, q_items)
    rf = select(r_type, r_items)
    scores = pair_logits(model, q_type, r_type, qf, rf).numpy()
    rel = relevance(q_items, r_items)

    return p_at_1(scores, rel), r_at_1(scores, rel), m_ap(scores, rel), len(q_items), len(r_items)


def folds_for_dataset(dataset: str) -> list[int]:
    if dataset == "esc50":
        return [1, 2, 3, 4, 5]
    if dataset == "us8k":
        return list(range(1, 11))
    return []


def main() -> None:
    args = parse_args()
    if args.query_type == args.result_type:
        raise ValueError("query-type and result-type must be different")

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_path = resolve_checkpoint_path(model_path)

    device = torch.device(args.device)
    model = AudioCLIP(pretrained=str(model_path)).to(device)
    model.eval()

    text_items, audio_items, image_items = build_items(args.dataset, dataset_root)
    all_items = {"text": text_items, "audio": audio_items, "image": image_items}

    audio_len = default_audio_length(args.dataset) if args.audio_length is None else args.audio_length
    sample_rate = 44_100

    use_paper_folds = args.protocol == "paper" and args.dataset in {"esc50", "us8k"}
    if use_paper_folds:
        folds = [args.fold] if args.fold is not None else folds_for_dataset(args.dataset)
        if not folds:
            raise ValueError(f"No folds available for dataset {args.dataset}")

        fold_metrics: list[tuple[float, float, float, int, int]] = []
        for fold in folds:
            fold_audio = [it for it in audio_items if it.fold == fold]
            if not fold_audio:
                continue

            fold_labels = sorted({next(iter(it.labels)) for it in fold_audio})
            fold_text = [Item(key=lb, labels={lb}, text=lb) for lb in fold_labels]

            fold_items = {
                "text": fold_text if args.query_type == "text" or args.result_type == "text" else [],
                "audio": fold_audio if args.query_type == "audio" or args.result_type == "audio" else [],
                "image": image_items,
            }

            p1, r1, mp, nq, nr = evaluate_once(
                model,
                args.query_type,
                args.result_type,
                fold_items,
                args.batch_size,
                device,
                sample_rate,
                audio_len,
                args.prompt_template,
            )
            fold_metrics.append((p1, r1, mp, nq, nr))

        if not fold_metrics:
            raise RuntimeError("No fold metrics computed. Check dataset root and modality choices.")

        p1 = float(np.mean([m[0] for m in fold_metrics]))
        r1 = float(np.mean([m[1] for m in fold_metrics]))
        map_score = float(np.mean([m[2] for m in fold_metrics]))
        nq = int(np.mean([m[3] for m in fold_metrics]))
        nr = int(np.mean([m[4] for m in fold_metrics]))
        mode_info = f"paper-fold-wise ({len(fold_metrics)} folds)"
    else:
        p1, r1, map_score, nq, nr = evaluate_once(
            model,
            args.query_type,
            args.result_type,
            all_items,
            args.batch_size,
            device,
            sample_rate,
            audio_len,
            args.prompt_template,
        )
        mode_info = "global"

    model_label = "audio-head" if "partial" in model_path.name.lower() else "full-model"

    print("Table 4 setting")
    print(f"  dataset: {args.dataset}")
    print(f"  query-result: {args.query_type} -> {args.result_type}")
    print(f"  protocol: {mode_info}")
    print(f"  model: {model_path}")
    print(f"  model-label: {model_label}")
    print(f"  queries(avg): {nq}")
    print(f"  results(avg): {nr}")
    print()
    print("Scores")
    print(f"  P@1: {p1 * 100.0:.2f}")
    print(f"  R@1: {r1 * 100.0:.2f}")
    if math.isnan(map_score):
        print("  mAP: N/A")
    else:
        print(f"  mAP: {map_score * 100.0:.2f}")


if __name__ == "__main__":
    main()
