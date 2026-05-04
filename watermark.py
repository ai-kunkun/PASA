from __future__ import annotations

import hashlib
import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F

try:
    from tokenizers import Tokenizer
except Exception:  
    Tokenizer = object


class Watermark:
    """K-cluster semantic watermarking for text generation and detection."""

    def __init__(
        self,
        device: torch.device,
        watermark_tokenizer: Tokenizer,
        watermark_model,
        auxilary_tokenizer: Tokenizer,
        auxilary_model,
        key_map_path: str,
        num_semantic_keys: int = 50,
        alpha: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens: int = 300,
        min_new_tokens: int = 200,
        key: int = 123,
        T: float = 1.0,
        start: int = 5,
        window_size: int = 2,
        verbose: bool = False,
    ):
        self.device = device
        self.watermark_tokenizer = watermark_tokenizer
        self.watermark_model = watermark_model
        self.auxilary_tokenizer = auxilary_tokenizer
        self.auxilary_model = auxilary_model

        self.num_semantic_keys = int(num_semantic_keys)
        self.alpha = float(alpha)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.max_new_tokens = int(max_new_tokens)
        self.min_new_tokens = int(min_new_tokens)
        self.key = int(key)
        self.T = float(T)
        self.start = int(start)
        self.window_size = int(window_size)
        self.verbose = bool(verbose)

        if self.num_semantic_keys <= 0:
            raise ValueError("num_semantic_keys must be positive")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if not os.path.exists(key_map_path):
            raise FileNotFoundError(f"key_map_path does not exist: {key_map_path}")

        self._map_cache: dict[int, torch.Tensor] = {}
        self.token_to_key_map = self._load_token_to_key_map(key_map_path)

    @staticmethod
    def _as_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x

    def _sanitize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        weights = self._as_batch(weights)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)
        bad_rows = weights.sum(dim=-1) <= 0
        if bad_rows.any():
            weights = weights.clone()
            weights[bad_rows] = 1.0
        return weights

    def _normalize_probs(self, probs: torch.Tensor) -> torch.Tensor:
        probs = self._sanitize_weights(probs)
        return probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)

    def _sample_one(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(self._sanitize_weights(weights), num_samples=1)

    def _load_token_to_key_map(self, path: str) -> torch.Tensor:
        df = pd.read_csv(path)
        required = {"token_id", "best_cluster_idx"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns {required}; got {set(df.columns)}")

        vocab_size = max(
            int(getattr(self.auxilary_tokenizer, "vocab_size", 0) or 0),
            int(len(self.auxilary_tokenizer)) if hasattr(self.auxilary_tokenizer, "__len__") else 0,
            int(getattr(getattr(self.auxilary_model, "config", None), "vocab_size", 0) or 0),
        )
        if vocab_size <= 0:
            raise ValueError("Cannot infer vocabulary size from auxiliary tokenizer/model")

        K = self.num_semantic_keys
        df = df[(df["token_id"] >= 0) & (df["token_id"] < vocab_size)]
        df = df[(df["best_cluster_idx"] >= 0) & (df["best_cluster_idx"] < K)]
        if df.empty:
            raise ValueError("No valid token-to-cluster entries after filtering")

        mapping = torch.full((vocab_size,), -1, dtype=torch.long)
        mapping[torch.tensor(df["token_id"].values, dtype=torch.long)] = torch.tensor(
            df["best_cluster_idx"].values,
            dtype=torch.long,
        )
        mapping = mapping.to(self.device)

        mapped_count = int(((mapping >= 0) & (mapping < K)).sum().item())
        if mapped_count == 0:
            raise ValueError("Token-to-cluster mapping has zero coverage")
        if self.verbose:
            print(f"[Watermark] mapping coverage: {mapped_count}/{vocab_size} ({mapped_count / vocab_size:.2%})")
        return mapping

    def _mapping_for_vocab(self, vocab_size: int) -> torch.Tensor:
        vocab_size = int(vocab_size)
        if vocab_size <= 0:
            return self.token_to_key_map
        if vocab_size in self._map_cache:
            return self._map_cache[vocab_size]

        mapping = self.token_to_key_map
        current_size = int(mapping.size(0))
        if current_size == vocab_size:
            aligned = mapping
        elif current_size > vocab_size:
            aligned = mapping[:vocab_size]
        else:
            pad = torch.full((vocab_size - current_size,), -1, dtype=mapping.dtype, device=mapping.device)
            aligned = torch.cat([mapping, pad], dim=0)

        self._map_cache[vocab_size] = aligned
        return aligned

    def _calc_banned_ngram_tokens(
        self,
        prev_input_ids: torch.Tensor,
        no_repeat_ngram_size: int,
        cur_len: int,
    ) -> list[list[int]]:
        if no_repeat_ngram_size <= 0 or cur_len + 1 < no_repeat_ngram_size:
            return [[]]

        tokens = prev_input_ids[0].tolist()
        generated = {}
        for ngram in zip(*[tokens[i:] for i in range(no_repeat_ngram_size)]):
            generated.setdefault(tuple(ngram[:-1]), []).append(ngram[-1])

        start_idx = cur_len + 1 - no_repeat_ngram_size
        prefix = tuple(prev_input_ids[0, start_idx:cur_len].tolist())
        return [generated.get(prefix, [])]

    def _postprocess_next_token_scores(self, logits: torch.Tensor, prev_output_tokens: torch.Tensor, eos_token_id: int | None) -> None:
        if self.repetition_penalty != 1.0 and prev_output_tokens.numel() > 0:
            for token_id in set(prev_output_tokens[0].tolist()):
                if 0 <= token_id < logits.size(-1):
                    logits[0, token_id] = (
                        logits[0, token_id] * self.repetition_penalty
                        if logits[0, token_id] < 0
                        else logits[0, token_id] / self.repetition_penalty
                    )

        if eos_token_id is not None and prev_output_tokens.size(1) < self.min_new_tokens and 0 <= eos_token_id < logits.size(-1):
            logits[:, eos_token_id] = -float("inf")

        if self.no_repeat_ngram_size > 0 and prev_output_tokens.numel() > 0:
            banned = self._calc_banned_ngram_tokens(prev_output_tokens, self.no_repeat_ngram_size, prev_output_tokens.size(1))
            token_ids = [t for t in banned[0] if 0 <= t < logits.size(-1)]
            if token_ids:
                logits[0, token_ids] = -float("inf")

    def _top_p(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / max(float(self.T), 1e-12)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = self._normalize_probs(torch.softmax(logits, dim=-1))

        if self.top_p >= 1.0:
            return probs

        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > self.top_p] = 0.0
        sorted_probs = self._normalize_probs(sorted_probs)
        return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

    def _stopping_criteria(self, output_ids: torch.Tensor) -> bool:
        eos_token_id = getattr(self.watermark_tokenizer, "eos_token_id", None)
        return eos_token_id is not None and output_ids.size(1) >= self.min_new_tokens and output_ids[0, -1].item() == eos_token_id

    def _get_context_seed(self, context_ids: torch.Tensor) -> str:
        if context_ids.numel() == 0:
            return "empty"

        mapping = self._mapping_for_vocab(int(context_ids.max().item()) + 1)
        valid = (context_ids >= 0) & (context_ids < mapping.size(0))
        cluster_ids = torch.full_like(context_ids, -1)
        if valid.any():
            cluster_ids[valid] = mapping[context_ids[valid]]
        return "-".join(map(str, cluster_ids.tolist()))

    def _generate_U(self, context_seed: str, size: int) -> torch.Tensor:
        seed_input = f"{self.key}-{context_seed}".encode("utf-8")
        seed = int(hashlib.sha256(seed_input).hexdigest(), 16) % (2**32)
        generator = torch.Generator(device="cuda" if self.device.type == "cuda" else "cpu")
        generator.manual_seed(seed)
        return torch.rand(int(size), generator=generator, device=self.device).unsqueeze(0)

    def _gumbel_max_sample(self, probs: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        probs = self._normalize_probs(probs)
        U = torch.clamp(U, 1e-12, 1.0 - 1e-12)
        noise = -torch.log(-torch.log(U))
        return torch.argmax(torch.log(torch.clamp(probs, min=1e-20)) + noise, dim=-1).unsqueeze(0)

    def _aggregate_probs_to_keys(self, probs: torch.Tensor) -> torch.Tensor:
        probs = self._as_batch(probs)
        batch_size, vocab_size = probs.shape
        K = self.num_semantic_keys
        key_probs = torch.zeros(batch_size, K, device=self.device, dtype=probs.dtype)

        mapping = self._mapping_for_vocab(vocab_size)
        valid = (mapping >= 0) & (mapping < K)
        if not valid.any():
            return key_probs

        token_ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
        key_ids = mapping[token_ids]
        key_probs.scatter_add_(
            dim=1,
            index=key_ids.unsqueeze(0).expand(batch_size, -1),
            src=probs[:, token_ids],
        )
        return key_probs

    def _zeta_distribution(self, key_mass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_mass = self._as_batch(key_mass)
        cap = torch.tensor(self.alpha, device=self.device, dtype=key_mass.dtype)
        trunc = torch.minimum(key_mass, cap)
        overflow_per_key = torch.clamp(key_mass - cap, min=0.0)
        overflow_sum = overflow_per_key.sum(dim=-1, keepdim=True)
        zeta_probs = self._normalize_probs(torch.cat([trunc, overflow_sum], dim=-1))
        return zeta_probs, overflow_per_key, overflow_sum

    def _sample_token_in_key(self, probs: torch.Tensor, key_index: int) -> torch.Tensor:
        probs = self._as_batch(probs)
        mapping = self._mapping_for_vocab(probs.size(-1))
        token_ids = torch.nonzero(mapping == int(key_index), as_tuple=False).squeeze(1)
        if token_ids.numel() == 0:
            return self._sample_one(probs)
        local_idx = self._sample_one(probs[:, token_ids])
        return token_ids[local_idx.item()].view(1, 1)

    def _sample_token_given_tilde_zeta(
        self,
        probs: torch.Tensor,
        key_mass: torch.Tensor,
        overflow_per_key: torch.Tensor,
        overflow_sum: torch.Tensor,
    ) -> torch.Tensor:
        probs = self._as_batch(probs)
        key_mass = self._as_batch(key_mass)
        K = self.num_semantic_keys

        mapping = self._mapping_for_vocab(probs.size(-1))
        valid = (mapping >= 0) & (mapping < K)
        if not valid.any() or (overflow_sum <= 0).any():
            return self._sample_one(probs)

        token_ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
        key_ids = mapping[token_ids]
        q_tok = probs[:, token_ids]
        m_tok = torch.clamp(key_mass[:, key_ids], min=1e-12)
        overflow_tok = overflow_per_key[:, key_ids]
        weights = q_tok * overflow_tok / torch.clamp(m_tok * overflow_sum, min=1e-12)

        if self._sanitize_weights(weights).sum() <= 1e-12:
            weights = q_tok

        local_idx = self._sample_one(weights)
        return token_ids[local_idx.item()].view(1, 1)

    def _watermarking(self, probs: torch.Tensor, context_seed: str) -> Tuple[torch.Tensor, int]:
        probs = self._as_batch(probs)
        key_mass = self._aggregate_probs_to_keys(probs)
        zeta_probs, overflow_per_key, overflow_sum = self._zeta_distribution(key_mass)
        zeta = int(self._gumbel_max_sample(zeta_probs, self._generate_U(context_seed, self.num_semantic_keys + 1)).item())

        if zeta < self.num_semantic_keys:
            return self._sample_token_in_key(probs, zeta), zeta

        return self._sample_token_given_tilde_zeta(probs, key_mass, overflow_per_key, overflow_sum), zeta

    def _generate(self, prompt: str, watermarked: bool) -> str:
        input_ids = self.watermark_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids = torch.empty(1, 0, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        past_key_values = None

        for t in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.watermark_model(
                    input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]
            self._postprocess_next_token_scores(logits, output_ids, getattr(self.watermark_tokenizer, "eos_token_id", None))
            probs = self._top_p(logits)

            if watermarked and t >= self.start:
                window = min(t, self.window_size)
                context_ids = output_ids[0, -window:] if window > 0 else output_ids[0, :0]
                next_id, _ = self._watermarking(probs, self._get_context_seed(context_ids))
            else:
                next_id = self._sample_one(probs)

            input_ids = torch.cat([input_ids, next_id], dim=-1)
            output_ids = torch.cat([output_ids, next_id], dim=-1)
            past_key_values = outputs.past_key_values
            attention_mask = F.pad(attention_mask, (0, 1), "constant", 1)

            if self._stopping_criteria(output_ids):
                break

        return self.watermark_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_unwatermarked(self, prompt: str) -> str:
        return self._generate(prompt, watermarked=False)

    def generate_watermarked(self, prompt: str) -> str:
        return self._generate(prompt, watermarked=True)

    def detection(self, text: str) -> Tuple[float, int, int]:
        ids = self.auxilary_tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        seq_len = ids.size(1)
        if seq_len <= self.start:
            return 0.0, 0, 0

        with torch.no_grad():
            logits = self.auxilary_model(ids).logits

        mapping_aux = self._mapping_for_vocab(int(logits.size(-1)))
        hits = 0
        count = 0

        for t in range(self.start, seq_len):
            logits_t = logits[:, t - 1, :].clone()
            self._postprocess_next_token_scores(logits_t, ids[:, :t], getattr(self.auxilary_tokenizer, "eos_token_id", None))
            probs_t = self._top_p(logits_t)

            window = min(t, self.window_size)
            context_ids = ids[0, t - window : t] if window > 0 else ids[0, :0]
            key_mass = self._aggregate_probs_to_keys(probs_t)
            zeta_probs, _, _ = self._zeta_distribution(key_mass)
            predicted_zeta = int(self._gumbel_max_sample(zeta_probs, self._generate_U(self._get_context_seed(context_ids), self.num_semantic_keys + 1)).item())

            if predicted_zeta == self.num_semantic_keys:
                count += 1
                continue

            token_id = int(ids[0, t].item())
            if token_id >= mapping_aux.size(0):
                continue

            actual_key = int(mapping_aux[token_id].item())
            if actual_key < 0 or actual_key >= self.num_semantic_keys:
                continue

            count += 1
            hits += int(predicted_zeta == actual_key)

        return (hits / count if count > 0 else 0.0), hits, count
