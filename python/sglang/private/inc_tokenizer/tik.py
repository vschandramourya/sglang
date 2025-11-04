from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import tiktoken
from transformers import AutoTokenizer

try:
    from tokenizers import (
        Tokenizer as HFTokenizerFast,  # faster tokenizer.json load if present
    )
except Exception:
    HFTokenizerFast = None

# LLaMA/DeepSeek-style Unicode regex (close to HF pretokenization)
_LLAMA_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_MARKER_RE = re.compile(r"^<｜[^｜]+｜>$")  # matches <｜...｜> exactly


def _build_byte_decoder() -> Dict[str, int]:
    """GPT-2 reversible bytes→unicode map (glyph -> raw byte)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _collect_deepseek_specials(
    tok_json: dict, vocab: Dict[str, int]
) -> tuple[Dict[str, int], Set[str]]:
    """
    Ultra-simple DeepSeek specials collector.
    Promote to special if:
      - added_token.special is True
      - OR added_token.normalized is True
      - OR content matches <｜...｜>
    Also sweep vocab for <｜...｜> forms.
    """
    special_tokens: Dict[str, int] = {}
    special_strings: Set[str] = set()

    for t in tok_json.get("added_tokens", []):
        s = t.get("content")
        tid = t.get("id")
        if s is None or tid is None:
            continue
        if (
            t.get("special", False)
            or t.get("normalized", False)
            or _MARKER_RE.fullmatch(s)
        ):
            special_tokens[s] = tid
            special_strings.add(s)

    for s, tid in vocab.items():
        if s in special_strings:
            continue
        if _MARKER_RE.fullmatch(s):
            special_tokens[s] = tid
            special_strings.add(s)

    # Common fallbacks that some repos put in vocab (rare here, harmless)
    for sym in ("<s>", "</s>", "<unk>"):
        if sym in vocab and sym not in special_tokens:
            special_tokens[sym] = vocab[sym]
            special_strings.add(sym)

    return special_tokens, special_strings


class TiktokenWrapper:
    """
    Minimal HF-compatible wrapper around tiktoken for BPE tokenizers like DeepSeek-V3.
      - Exact special token IDs (including 'normalized' or marker-shaped ones)
      - Correct merge priorities (use HF ids as ranks)
      - BOS/EOS parity with HF config
      - Fast encode/decode; reuse HF for chat template
    """

    def __init__(self, tokenizer_path: str):
        self.path = Path(tokenizer_path)
        self.hf = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, local_files_only=True
        )

        # --- Load tokenizer.json (+ optional tokenizer_config.json) ---
        tj_path = self.path / "tokenizer.json"
        if HFTokenizerFast is not None:
            tok_json = json.loads(HFTokenizerFast.from_file(str(tj_path)).to_str())
        else:
            tok_json = json.load(open(tj_path, "r", encoding="utf-8"))

        cfg_path = self.path / "tokenizer_config.json"
        tok_cfg = (
            json.load(open(cfg_path, "r", encoding="utf-8"))
            if cfg_path.exists()
            else {}
        )

        model = tok_json["model"]
        if model["type"] != "BPE":
            raise ValueError(f"Expected BPE tokenizer; got {model['type']}")

        vocab: Dict[str, int] = model["vocab"]

        # --- Specials (ultra-simple rules) ---
        special_tokens, special_strings = _collect_deepseek_specials(tok_json, vocab)

        # --- Build mergeable_ranks EXCLUDING specials ---
        byte_decoder = _build_byte_decoder()

        def to_bytes_gpt2glyph(s: str) -> bytes:
            return bytes(byte_decoder[ch] for ch in s)

        mergeable_ranks: Dict[bytes, int] = {}
        for tok_str, tok_id in vocab.items():
            if tok_str in special_strings:
                continue  # do NOT merge specials
            try:
                b = to_bytes_gpt2glyph(tok_str)
            except KeyError:
                # fallback: utf-8 if a token isn't in the GPT-2 glyph table
                b = tok_str.encode("utf-8")
            mergeable_ranks[b] = tok_id

        # --- tiktoken Encoding ---
        self.encoding = tiktoken.Encoding(
            name="deepseek_v3_tiktoken",
            pat_str=_LLAMA_PAT,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        # --- Cache special ids/strings for encode/decode helpers ---
        self.special_strings: Set[str] = special_strings  # set[str]
        self._special_id_to_str = {tid: s for s, tid in special_tokens.items()}
        self._special_ids: Set[int] = set(self._special_id_to_str.keys())

        # --- BOS/EOS behavior (DeepSeek-V3: add_bos_token=True, add_eos_token=False) ---
        self.add_bos_token: bool = bool(tok_cfg.get("add_bos_token", False))
        self.add_eos_token: bool = bool(tok_cfg.get("add_eos_token", False))
        self.bos_id: Optional[int] = special_tokens.get("<｜begin▁of▁sentence｜>")
        self.eos_id: Optional[int] = special_tokens.get("<｜end▁of▁sentence｜>")

        # --- Misc for compatibility ---
        self.vocab_size = len(vocab)
        self.tokenizer = self.hf  # delegate unknown attrs

    # ---------------- Public API ----------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        text_pair: Optional[str] = None,
        **kwargs,
    ) -> List[int]:
        """
        HF-compatible encode. If your text includes literal DeepSeek specials
        (e.g., from apply_chat_template), they will be kept atomic.
        """
        allowed_special: Set[str] = self.special_strings  # must be a set

        if text_pair is None:
            ids = self.encoding.encode(text, allowed_special=allowed_special)
            if add_special_tokens:
                # Mirror HF add_bos_token/add_eos_token
                if (
                    self.add_bos_token
                    and self.bos_id is not None
                    and (not ids or ids[0] != self.bos_id)
                ):
                    ids = [self.bos_id] + ids
                if (
                    self.add_eos_token
                    and self.eos_id is not None
                    and (not ids or ids[-1] != self.eos_id)
                ):
                    ids = ids + [self.eos_id]
            return ids

        # Pair encoding
        ids_a = self.encoding.encode(text, allowed_special=allowed_special)
        ids_b = self.encoding.encode(text_pair, allowed_special=allowed_special)
        if add_special_tokens:
            return self.hf.build_inputs_with_special_tokens(ids_a, ids_b)
        return ids_a + ids_b

    def decode(
        self,
        tokens: List[int],
        skip_special_tokens: bool = False,
        keep: Optional[Set[int]] = None,
        **kwargs,  # ignored
    ) -> str:
        """
        tiktoken decode with optional special-stripping (HF-like).
        """
        if not skip_special_tokens:
            return self.encoding.decode(tokens)
        keep = keep or set()
        filtered = [t for t in tokens if (t not in self._special_ids) or (t in keep)]
        return self.encoding.decode(filtered)

    def apply_chat_template(self, *args, **kwargs) -> str:
        # Use HF’s exact template rendering
        return self.hf.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate remaining attributes to the HF tokenizer."""
        if "tokenizer" not in self.__dict__:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.tokenizer, name)

    def __len__(self):
        return self.vocab_size


def create_tiktoken_tokenizer(tokenizer_path: str) -> TiktokenWrapper:
    return TiktokenWrapper(tokenizer_path)


def get_fast_tokenizer(path: str) -> TiktokenWrapper:
    return create_tiktoken_tokenizer(path)
