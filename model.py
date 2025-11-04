import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_angles(theeta, dim, seq_len):
    pos = 1/theeta**(torch.arange(0, dim, 2, device=device).float()/dim)
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, pos)
    unit_vecs = torch.polar(torch.ones_like(freqs), freqs)
    return unit_vecs

def brodcast(unit_vecs, x):
    assert unit_vecs.shape == (x.shape[1], x.shape[-1])
    n_dim = x.ndim
    shape = [d if i == 1 or i == n_dim-1 else 1 for i,d in enumerate(x.shape)]
    return unit_vecs.view(*shape)

def RoPE(W_Q, W_K, unit_vecs):
    complex_W_Q = torch.view_as_complex(W_Q.float().reshape(*W_Q.shape[:-1], -1, 2))
    complex_W_K = torch.view_as_complex(W_K.float().reshape(*W_K.shape[:-1], -1, 2))
    # print(complex_W_Q.shape)
    pos = brodcast(unit_vecs, complex_W_K)
    embedded_W_Q = torch.view_as_real(complex_W_Q * pos).float().flatten(3)
    embedded_W_K = torch.view_as_real(complex_W_K * pos).float().flatten(3)
    return embedded_W_Q, embedded_W_K

    
class Attention(nn.Module):
    def __init__(self, dim, n_heads, head_dim, n_kv_heads, n_kv_heads_reps, max_batch_size, max_seq_len):
        super().__init__()
        self.W_Q = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.W_K = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.W_V = nn.Linear(dim, n_kv_heads * head_dim, bias=False)

        self.register_buffer('CACHE_K', torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim))
        )
        self.register_buffer('CACHE_V', torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim))
        )

        self.wo = nn.Linear(dim, dim)

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.n_kv_heads_reps = n_kv_heads_reps


    def forward(self,x, freq=None, start_pos=0, mask=None):
        bhz, seq_len, _ = x.shape

        query = self.W_Q(x).view(bhz, seq_len, self.n_heads, self.head_dim)
        key = self.W_K(x).view(bhz, seq_len, self.n_kv_heads, self.head_dim)
        value = self.W_V(x).view(bhz, seq_len, self.n_kv_heads, self.head_dim)

        query, key = RoPE(query, key, freq)

        self.CACHE_K[:bhz, start_pos:start_pos+seq_len] = key
        self.CACHE_V[:bhz, start_pos:start_pos+seq_len] = value

        keys = self.CACHE_K[:bhz, :start_pos+seq_len]
        values = self.CACHE_V[:bhz, :start_pos+seq_len]

        keys = torch.repeat_interleave(input=keys, repeats=self.n_kv_heads_reps, dim=-2)
        values = torch.repeat_interleave(input=values, repeats=self.n_kv_heads_reps, dim=-2)

        queries = query.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        
        out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask)
        out = out.transpose(1,2).contiguous().view(bhz, seq_len, -1)

        return self.wo(out)

    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, norm_eps):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
    
    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()

        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Transformer_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Attention_Norm = RMSNorm(dim=config["DIM"], norm_eps=config["NORM_EPS"])
        self.FFN_Norm = RMSNorm(dim=config["DIM"], norm_eps=config["NORM_EPS"])
        self.Attention = Attention(dim=config["DIM"],
                                   n_heads=config["N_HEADS"],
                                   head_dim=config["HEAD_DIM"],
                                   n_kv_heads=config["N_KV_HEADS"],
                                   n_kv_heads_reps=config["N_KV_HEAD_REP"],
                                   max_batch_size=config["MAX_BATCH_SIZE"],
                                   max_seq_len=config["MAX_SEQ_LEN"])
        self.FeedForward = FeedForward(dim=config["DIM"],
                                       ffn_dim=config["FFN_DIM"])
    def forward(self, x, freq, start_pos, mask):
        shortcut = x
        x = self.Attention_Norm(x)
        x = self.Attention(x, freq, start_pos, mask)
        x = x + shortcut

        shortcut = x
        x = self.FFN_Norm(x)
        x = self.FeedForward(x)
        x = x + shortcut

        return x


class BLUE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_embedding = nn.Embedding(config["VOCAB_SIZE"], config["DIM"])
        self.layers = nn.ModuleList()
        for _ in range(config["N_LAYERS"]):
            self.layers.append(Transformer_Block(config))
        self.norm = RMSNorm(config["DIM"], config["NORM_EPS"])
        self.output = nn.Linear(config["DIM"], config["VOCAB_SIZE"], bias=False)

        self.register_buffer(
            'freq',
            calculate_angles(
                config["ROPE_THETA"],
                config["HEAD_DIM"],
                config["MAX_SEQ_LEN"] * 2
            )
        )
    
    def reset_cache(self):
        for name, module in self.named_modules():
            if hasattr(module, "CACHE_K"):
                module.CACHE_K.zero_()
            if hasattr(module, "CACHE_V"):
                module.CACHE_V.zero_()

    
    def forward(self, tokens, start_pos):
        bhz, seq_len = tokens.shape
        x = self.tok_embedding(tokens)
        freq = self.freq[start_pos : start_pos+seq_len]

        mask = None
        if seq_len > 1:
            total_len = start_pos + seq_len
            mask = torch.full((seq_len, total_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1 + start_pos)

        for layer in self.layers:
            x = layer(x, freq, start_pos, mask )
        x = self.norm(x)
        x = self.output(x).float()
        
        return x


class Tokenizer:
    no_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path):
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, self.no_reserved_special_tokens - 5)]
        
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )
        self.n_words = self.model.n_vocab
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
        ):
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000
        MAX_NO_WHITESPACES_CHARS = 25_000
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t
    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(cast(List[int], t))

    def _split_whitespaces_or_nonwhitespaces(self,
        s: str, max_consecutive_slice_len: int
    ):
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]     

