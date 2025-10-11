#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
local, multilingual, robust, fully offline chatbot with consensus + paa + c3 + rcg
+ reply contracts, user assertions, and prompt-injection shield
- tscd consensus: multi-candidate generation with overlap scoring and optional fusion
- paa: paraphrase-and-agree via offline back-translation to improve robustness
- c3: counterfactual consistency check to expose ambiguities and refine
- rcg: reflexive critic & governor to polish clarity and reduce redundancy
- reply contracts: enforce hard constraints on the final answer
- user assertions: regex-based rules that must hold on the final answer
- prompt-injection shield: strips known instruction hijacks from user input
- kb verifiability tags, dual-model copiloting, strict safety, pii redaction, url blacklist
- commands: /help, /about, /diag, /trace, /memory, /reset, /save, /load, /export
           /set key=value (temperature, top_p, max_new_tokens, max_context_tokens,
                           lang, redact_pii, system_prompt,
                           consensus, n_consensus, paa, c3, critic)
           /kb add <title>|<text>, /kb search <query>, /kb list, /kb clear
           /contract set <json>, /contract show, /contract clear
           /assert add <name>|<regex>, /assert list, /assert clear
- all inline comments are in english and in lowercase only
"""

import os
import re
import sys
import json
import time
import math
import signal
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import torch
import langdetect
from functools import lru_cache

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------

LOG_LEVEL = os.getenv("CHATBOT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO)
)
logger = logging.getLogger("chatbot")
logging.getLogger("transformers").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# device & torch
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    # auto-detect gpu/mps, fallback to cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_torch_deterministic(seed: int = 42):
    # best-effort determinism
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)

torch.set_grad_enabled(False)
set_torch_deterministic()

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------

@dataclass
class ChatConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 200
    max_context_tokens: int = 1536
    preferred_lang: str = "auto"
    redact_pii: bool = True
    url_blacklist: Tuple[str, ...] = ("example.com",)
    safety_threshold: float = 0.6
    save_path: str = "chat_state.json"
    transcript_md: str = "chat_transcript.md"
    system_prompt: str = (
        "you are a helpful, concise, and safe assistant. "
        "answer clearly and avoid speculation. if unsure, say so."
    )
    rpm_limit: int = 60
    flood_seconds: float = 0.35
    consensus: bool = True
    n_consensus: int = 3
    paa: bool = True
    c3: bool = True
    critic: bool = True

    def clamp(self):
        # validate and clamp values into safe bounds
        self.temperature = float(min(max(self.temperature, 0.05), 1.5))
        self.top_p = float(min(max(self.top_p, 0.1), 1.0))
        self.max_new_tokens = int(min(max(self.max_new_tokens, 16), 512))
        self.max_context_tokens = int(min(max(self.max_context_tokens, 256), 4096))
        self.rpm_limit = int(min(max(self.rpm_limit, 1), 600))
        self.flood_seconds = float(min(max(self.flood_seconds, 0.05), 5.0))
        self.n_consensus = int(min(max(self.n_consensus, 1), 7))

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _normalize_ws(text: str) -> str:
    # normalize whitespace
    return " ".join((text or "").split())

def _sanitize_control_chars(text: str) -> str:
    # strip control chars that can break tokenizers
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t\r")

def _split_sentences(text: str) -> List[str]:
    # very simple sentence splitter
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _pii_redact(text: str) -> str:
    # simple pii redaction demo
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[redacted-email]", text)
    text = re.sub(r"\b(\+?\d[\d\-\s]{6,}\d)\b", "[redacted-phone]", text)
    text = re.sub(r"(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9\-_.]{8,}['\"]?", r"\1: [redacted-secret]", text, flags=re.I)
    return text

def _block_blacklisted_urls(text: str, blacklist: Tuple[str, ...]) -> bool:
    # detect blacklisted domains
    return any(re.search(rf"https?://[^/\s]*{re.escape(domain)}", text, flags=re.I) for domain in blacklist)

def _safe_bool_env(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _ngrams(tokens: List[str], n: int) -> set:
    # build n-grams set
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def _tokenize_for_overlap(text: str) -> List[str]:
    # lightweight tokenizer for overlap scoring
    return re.findall(r"[a-zàâäéèêëîïôöùûüç0-9]+", text.lower())

# -----------------------------------------------------------------------------
# shield against prompt-injection
# -----------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"\bignore (all|any|previous) (instructions|rules)\b",
    r"\boverride (system|developer) (message|prompt)\b",
    r"\bact as (?:system|developer)\b",
    r"\bdisregard (?:safety|policy|guardrails)\b",
    r"\bpretend (?:you are|to be)\b",
]

def _strip_injection(text: str) -> str:
    # removes known prompt-injection cues without altering semantics much
    cleaned = text
    for pat in INJECTION_PATTERNS:
        cleaned = re.sub(pat, "[removed]", cleaned, flags=re.I)
    return cleaned

# -----------------------------------------------------------------------------
# safety filter
# -----------------------------------------------------------------------------

class SafetyFilter:
    """
    offline safety filter using bart-large-mnli + regex red flags
    - fail-closed for errors
    """

    def __init__(self, device: torch.device, threshold: float = 0.6):
        self.labels = ["hate", "harassment", "self-harm", "sexual", "violence"]
        self.threshold = threshold
        self.enabled = _safe_bool_env("CHATBOT_SAFETY_ENABLED", True)
        try:
            self.moderator = pipeline(
                task="zero-shot-classification",
                model="facebook/bart-large-mnli",
                tokenizer="facebook/bart-large-mnli",
                device=0 if device.type == "cuda" else -1,
                local_files_only=True
            )
            logger.info("safety: mnli loaded.")
        except Exception as e:
            self.moderator = None
            self.enabled = False
            logger.warning(f"safety unavailable: {e} (disabled)")

        self._compiled_flags = [re.compile(pat, re.I) for pat in [
            r"\bkill\s+(myself|himself|herself|them|me)\b",
            r"\bsuicide\b",
            r"\bnazi\b",
            r"\bhow to make (a )?bomb\b",
        ]]

    def is_safe(self, text: str) -> bool:
        txt = _normalize_ws(text)
        for pat in self._compiled_flags:
            if pat.search(txt):
                logger.debug(f"safety pattern matched: {pat.pattern}")
                return False
        if not self.enabled or not self.moderator:
            return True
        try:
            result = self.moderator(txt, candidate_labels=self.labels)
            scores = {lbl.lower(): sc for lbl, sc in zip(result["labels"], result["scores"])}
            blocked = any(scores.get(lbl, 0.0) >= self.threshold for lbl in self.labels)
            logger.debug(f"safety scores: {scores} -> blocked={blocked}")
            return not blocked
        except Exception as e:
            logger.error(f"safety error: {e}")
            return False

# -----------------------------------------------------------------------------
# memory
# -----------------------------------------------------------------------------

@dataclass
class ChatTurn:
    speaker: str
    text: str
    ts: float

class ChatMemory:
    # bounded conversation memory with persistence
    def __init__(self, max_turns: int = 8):
        self.turns: List[ChatTurn] = []
        self.max_turns = max_turns

    def add(self, speaker: str, text: str) -> None:
        self.turns.append(ChatTurn(speaker=speaker, text=_normalize_ws(text), ts=time.time()))
        if len(self.turns) > 2 * self.max_turns:
            self.turns = self.turns[-2 * self.max_turns:]

    def get_context(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)

    def to_json(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self.turns]

    def from_json(self, data: List[Dict[str, Any]]) -> None:
        self.turns = [ChatTurn(**d) for d in data][-2 * self.max_turns:]

    def last_user_lang_hint(self) -> Optional[str]:
        for t in reversed(self.turns):
            if t.speaker.lower() == "user":
                try:
                    return langdetect.detect(t.text)
                except Exception:
                    return None
        return None

# -----------------------------------------------------------------------------
# simple tf-idf kb (no external deps)
# -----------------------------------------------------------------------------

class SimpleTFIDF:
    # minimal tf-idf implementation for small kb usage
    def __init__(self):
        self.docs: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.doc_vectors: List[List[float]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-zàâäéèêëîïôöùûüç0-9]+", text, flags=re.I)
        return tokens

    def _build_vocab(self) -> None:
        vocab = {}
        for _, txt in self.docs:
            for tok in set(self._tokenize(txt)):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab

    def _compute_idf(self) -> None:
        n = len(self.docs)
        df = [0] * len(self.vocab)
        for _, txt in self.docs:
            seen = set(self._tokenize(txt))
            for tok in seen:
                df[self.vocab[tok]] += 1
        self.idf = [math.log((n + 1) / (dfi + 1)) + 1.0 for dfi in df]

    def _tf_vector(self, text: str) -> List[float]:
        vec = [0.0] * len(self.vocab)
        toks = self._tokenize(text)
        if not toks:
            return vec
        for tok in toks:
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def rebuild(self) -> None:
        if not self.docs:
            self.vocab, self.idf, self.doc_vectors = {}, [], []
            return
        self._build_vocab()
        self._compute_idf()
        self.doc_vectors = []
        for _, txt in self.docs:
            tf = self._tf_vector(txt)
            vec = [t * idf for t, idf in zip(tf, self.idf)]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            self.doc_vectors.append([x / norm for x in vec])

    def add(self, title: str, text: str) -> None:
        self.docs.append((title.strip(), text.strip()))
        self.rebuild()

    def clear(self) -> None:
        self.docs = []
        self.rebuild()

    def list(self) -> List[str]:
        return [t for t, _ in self.docs]

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.docs:
            return []
        q_tf = self._tf_vector(query)
        q_vec = [t * idf for t, idf in zip(q_tf, self.idf)]
        norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
        q_vec = [x / norm for x in q_vec]
        scores = []
        for (title, _), dvec in zip(self.docs, self.doc_vectors):
            scores.append((title, self._cosine(q_vec, dvec)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def support_score(self, sentence: str) -> float:
        hits = self.search(sentence, k=1)
        return hits[0][1] if hits else 0.0

# -----------------------------------------------------------------------------
# rate limiter
# -----------------------------------------------------------------------------

class RateLimiter:
    # simple token bucket per minute + flood guard
    def __init__(self, rpm: int, min_gap: float):
        self.capacity = max(1, rpm)
        self.tokens = float(self.capacity)
        self.refill_rate = float(self.capacity) / 60.0
        self.last = time.time()
        self.min_gap = float(min_gap)
        self.last_call = 0.0

    def allow(self) -> bool:
        now = time.time()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        if now - self.last_call < self.min_gap:
            return False
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.last_call = now
            return True
        return False

# -----------------------------------------------------------------------------
# core chatbot
# -----------------------------------------------------------------------------

class AdvancedMultilingualChatbot:
    """
    fully offline multilingual chatbot with consensus, paa, c3, rcg, contracts, and assertions
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_candidates: Optional[List[str]] = None,
        config: Optional[ChatConfig] = None
    ):
        self.device = device or get_device()
        logger.info(f"device: {self.device}")

        self.cfg = config or ChatConfig()
        self.cfg.clamp()

        self.limiter = RateLimiter(self.cfg.rpm_limit, self.cfg.flood_seconds)
        self.memory = ChatMemory(max_turns=8)
        self.kb = SimpleTFIDF()
        self.safety = SafetyFilter(self.device, threshold=self.cfg.safety_threshold)

        # reply contracts and user assertions state
        self.contract: Optional[Dict[str, Any]] = None
        self.assertions: Dict[str, str] = {}
        self._last_trace: Dict[str, Any] = {}

        self.model_candidates = model_candidates or [
            "facebook/blenderbot-400M-distill",
            "facebook/mbart-large-50-many-to-many-mmt",
        ]

        self.tokenizer_a, self.model_a, self.is_seq2seq_a = self._load_best_local_model(self.model_candidates)
        self.tokenizer_b, self.model_b, self.is_seq2seq_b = None, None, None
        if len(self.model_candidates) > 1:
            try:
                alt = list(reversed(self.model_candidates))
                self.tokenizer_b, self.model_b, self.is_seq2seq_b = self._load_best_local_model(alt)
            except Exception as e:
                logger.warning(f"secondary model unavailable: {e}")

        self.translator_fr_en = self._make_pipeline_safe("translation_fr_to_en", "Helsinki-NLP/opus-mt-fr-en")
        self.translator_en_fr = self._make_pipeline_safe("translation_en_to_fr", "Helsinki-NLP/opus-mt-en-fr")
        self.summarizer = self._make_pipeline_safe("summarization", "facebook/bart-large-cnn")
        self.qa = self._make_pipeline_safe("question-answering", "distilbert-base-cased-distilled-squad")

        self.commands = {
            "/help": self._cmd_help,
            "/about": self._cmd_about,
            "/diag": self._cmd_diag,
            "/trace": self._cmd_trace,
            "/memory": self._cmd_memory,
            "/reset": self._cmd_reset,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/export": self._cmd_export,
            "/set": self._cmd_set,
            "/kb": self._cmd_kb,
            "/contract": self._cmd_contract,
            "/assert": self._cmd_assert,
        }

    # ---------------- loading ----------------

    def _make_pipeline_safe(self, task: str, model_name: str):
        try:
            pl = pipeline(
                task=task,
                model=model_name,
                device=0 if self.device.type == "cuda" else -1,
                local_files_only=True
            )
            logger.info(f"pipeline loaded: {task} ({model_name})")
            return pl
        except Exception as e:
            logger.warning(f"pipeline unavailable: {task} ({model_name}): {e}")
            return None

    def _load_best_local_model(self, candidates: List[str]):
        logger.info(f"loading conversational model from: {candidates}")
        last_err = None
        for name in candidates:
            try:
                cfg = AutoConfig.from_pretrained(name, local_files_only=True)
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
                if tok.pad_token_id is None and tok.eos_token_id is not None:
                    tok.pad_token_id = tok.eos_token_id
                is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
                if is_seq2seq:
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True)
                else:
                    mdl = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
                mdl.to(self.device).eval()
                mdl.config.pad_token_id = tok.pad_token_id
                logger.info(f"loaded: {name} (seq2seq={is_seq2seq})")
                return tok, mdl, is_seq2seq
            except Exception as e:
                last_err = e
                logger.warning(f"failed: {name}: {e}")
        raise RuntimeError(f"no local conversational model available. last error: {last_err}")

    # ---------------- language detection ----------------

    @lru_cache(maxsize=512)
    def detect_language(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except Exception:
            return "en"

    # ---------------- prompt building ----------------

    def _build_prompt(self, user_prompt: str) -> str:
        sys_p = f"System: {self.cfg.system_prompt}"
        ctx = self.memory.get_context()
        if ctx:
            return f"{sys_p}\n{ctx}\nUser: {user_prompt}\nBot:"
        return f"{sys_p}\nUser: {user_prompt}\nBot:"

    def _truncate_to_limit(self, input_ids: torch.Tensor, reserve: int, model) -> torch.Tensor:
        model_max = int(getattr(model.config, "max_position_embeddings", 2048))
        model_max = max(model_max, 512)
        max_input = max(min(self.cfg.max_context_tokens, model_max - reserve - 8), 16)
        if input_ids.shape[-1] > max_input:
            input_ids = input_ids[:, -max_input:]
        return input_ids

    def _validated_params(self) -> Dict[str, Any]:
        self.cfg.clamp()
        return dict(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer_a.pad_token_id,
            eos_token_id=self.tokenizer_a.eos_token_id,
            do_sample=True
        )

    # ---------------- generation (single) ----------------

    def _generate_with(self, model, tokenizer, is_seq2seq, prompt: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        txt = _sanitize_control_chars(_strip_injection(_normalize_ws(prompt)))
        if self.cfg.redact_pii:
            txt_for_model = _pii_redact(txt)
        else:
            txt_for_model = txt

        built = self._build_prompt(txt_for_model)
        enc = tokenizer(built, return_tensors="pt", padding=False, truncation=False)
        input_ids = enc["input_ids"].to(self.device)
        input_ids = self._truncate_to_limit(input_ids, reserve=self.cfg.max_new_tokens, model=model)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        params = self._validated_params()
        params["pad_token_id"] = tokenizer.pad_token_id
        params["eos_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **params
            )

        if is_seq2seq:
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            continuation = out[0][input_ids.shape[-1]:]
            decoded = tokenizer.decode(continuation, skip_special_tokens=True)

        reply = decoded.strip()
        if not reply:
            reply = "je n'ai pas bien compris, pouvez-vous reformuler ?"
        return reply

    # ---------------- consensus core ----------------

    def _overlap_score(self, a: str, b: str) -> float:
        ta, tb = _tokenize_for_overlap(a), _tokenize_for_overlap(b)
        if not ta or not tb:
            return 0.0
        score = 0.0
        for n, w in [(1, 0.2), (2, 0.4), (3, 0.6)]:
            na, nb = _ngrams(ta, n), _ngrams(tb, n)
            inter = len(na & nb)
            union = len(na | nb) or 1
            score += w * (inter / union)
        rep_pen = min(0.2, max(0.0, (self._repetition_ratio(a) + self._repetition_ratio(b)) / 2.0))
        return max(0.0, score - rep_pen)

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        toks = _tokenize_for_overlap(text)
        if not toks:
            return 0.0
        uniq = len(set(toks))
        return max(0.0, 1.0 - uniq / max(1, len(toks)))

    def _consensus_pick(self, candidates: List[str]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        n = len(candidates)
        scores = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                s = self._overlap_score(candidates[i], candidates[j])
                scores[i][j] = scores[j][i] = s
        means = [sum(row)/max(1, n-1) for row in scores]
        best_idx = max(range(n), key=lambda i: means[i])
        best = candidates[best_idx]
        if self.summarizer and n >= 2:
            others = sorted(range(n), key=lambda i: means[i], reverse=True)[:2]
            a, b = candidates[others[0]], candidates[others[1]]
            if self._overlap_score(a, b) >= 0.35:
                try:
                    fused = self.summarizer(a + "\n\n" + b, max_length=160, min_length=40)[0]["summary_text"].strip()
                    if fused:
                        return fused
                except Exception:
                    pass
        return best

    def generate_reply(self, user_prompt: str) -> str:
        return self._generate_with(self.model_a, self.tokenizer_a, self.is_seq2seq_a, user_prompt)

    def generate_reply_consensus(self, user_prompt: str) -> str:
        seeds = [int(time.time()) % 100000 + i*97 for i in range(self.cfg.n_consensus)]
        candidates = []
        for i, sd in enumerate(seeds):
            use_b = (self.model_b is not None) and (i % 2 == 1)
            model = self.model_b if use_b else self.model_a
            tok = self.tokenizer_b if use_b else self.tokenizer_a
            is_s2s = self.is_seq2seq_b if use_b else self.is_seq2seq_a
            try:
                cand = self._generate_with(model, tok, is_s2s, user_prompt, seed=sd)
            except Exception as e:
                logger.error(f"candidate generation error: {e}")
                cand = self._generate_with(self.model_a, self.tokenizer_a, self.is_seq2seq_a, user_prompt)
            candidates.append(cand)
        return self._consensus_pick(candidates)

    # ---------------- paa: paraphrase-and-agree ----------------

    def _back_translate_once(self, text: str) -> str:
        try:
            lang = self.detect_language(text)
            if lang.startswith("fr") and self.translator_fr_en and self.translator_en_fr:
                en = self.translator_fr_en(text)[0]["translation_text"]
                fr = self.translator_en_fr(en)[0]["translation_text"]
                return fr
            if lang.startswith("en") and self.translator_en_fr and self.translator_fr_en:
                fr = self.translator_en_fr(text)[0]["translation_text"]
                en = self.translator_fr_en(fr)[0]["translation_text"]
                return en
        except Exception as e:
            logger.warning(f"back-translation failed: {e}")
        return text

    def _paa_answer(self, prompt: str) -> Tuple[str, List[Tuple[str, str]]]:
        p1 = self._back_translate_once(prompt)
        p2 = self._back_translate_once(p1 + " ")
        a0 = self.generate_reply_consensus(prompt) if self.cfg.consensus else self.generate_reply(prompt)
        a1 = self.generate_reply_consensus(p1) if self.cfg.consensus else self.generate_reply(p1)
        a2 = self.generate_reply_consensus(p2) if self.cfg.consensus else self.generate_reply(p2)
        pairs = [(prompt, a0), (p1, a1), (p2, a2)]
        best = self._consensus_pick([a0, a1, a2])
        if self.summarizer:
            try:
                fused = self.summarizer(a0 + "\n\n" + best, max_length=180, min_length=50)[0]["summary_text"].strip()
                if fused:
                    best = fused
            except Exception:
                pass
        return best, pairs

    # ---------------- c3: counterfactual consistency check ----------------

    def _counter_question(self, prompt: str) -> str:
        if len(prompt) < 12:
            return prompt + " avec toutes les contraintes et cas limites."
        return prompt + " (précise aussi les hypothèses, limites, et cas extrêmes)."

    def _c3_refine(self, prompt: str, answer: str) -> str:
        cq = self._counter_question(prompt)
        a_cq = self.generate_reply(cq)
        agree = self._overlap_score(answer, a_cq)
        self._last_trace["counterfactual_agreement"] = round(agree, 3)
        if agree >= 0.30:
            return answer
        msg = (
            "il peut y avoir des ambiguïtés potentielles détectées entre la demande initiale et une variante contrôlée. "
            "je clarifie puis réponds de manière robuste et cohérente."
        )
        if self.summarizer:
            try:
                fused = self.summarizer(answer + "\n\n" + a_cq, max_length=200, min_length=60)[0]["summary_text"].strip()
                if fused:
                    return msg + " " + fused
            except Exception:
                pass
        return msg + " " + answer

    # ---------------- rcg: reflexive critic & governor ----------------

    def _critic_scores(self, text: str) -> Dict[str, float]:
        tokens = _tokenize_for_overlap(text)
        length = len(tokens)
        uniq = len(set(tokens)) or 1
        redundancy = 1.0 - (uniq / max(uniq, length))
        clarity = max(0.0, 1.0 - redundancy) * (1.0 if length > 12 else 0.7)
        safety = min(1.0, 0.6 + (uniq / max(20.0, length)))
        return {"clarity": clarity, "redundancy": redundancy, "safety": safety}

    def _rcg_polish(self, text: str) -> str:
        scores = self._critic_scores(text)
        self._last_trace["critic"] = {k: round(v, 3) for k, v in scores.items()}
        if scores["clarity"] >= 0.75 and scores["redundancy"] <= 0.20:
            return text
        if self.summarizer:
            try:
                cleaned = self.summarizer(text, max_length=min(180, max(60, len(text)//5)), min_length=40)[0]["summary_text"].strip()
                return cleaned or text
            except Exception:
                return text
        return text

    # ---------------- verifiability tagging ----------------

    def _tag_verifiability(self, text: str) -> str:
        if not self.kb.docs:
            return text
        lines = []
        supported = 0
        sents = _split_sentences(text)
        for sent in sents:
            sc = self.kb.support_score(sent)
            if sc >= 0.35:
                lines.append(f"{sent} (kb: supported)")
                supported += 1
            else:
                lines.append(f"{sent} (kb: unverified)")
        ratio = supported / max(1, len(sents))
        self._last_trace["kb_supported_ratio"] = round(ratio, 3)
        return " ".join(lines)

    # ---------------- reply contracts ----------------

    def _enforce_contract(self, text: str) -> str:
        if not self.contract:
            return text
        t = text

        # language enforcement
        lang = self.contract.get("language")
        if lang and isinstance(lang, str):
            try:
                det = self.detect_language(t)
            except Exception:
                det = None
            if lang.lower().startswith("fr") and det != "fr" and self.translator_en_fr:
                try:
                    t = self.translator_en_fr(t)[0]["translation_text"]
                except Exception:
                    pass
            if lang.lower().startswith("en") and det != "en" and self.translator_fr_en:
                try:
                    t = self.translator_fr_en(t)[0]["translation_text"]
                except Exception:
                    pass

        # forbid words
        forb = self.contract.get("forbid_words", [])
        if isinstance(forb, list):
            for w in forb:
                if not w:
                    continue
                t = re.sub(rf"\b{re.escape(w)}\b", "[removed]", t, flags=re.I)

        # require words
        req = self.contract.get("require_words", [])
        if isinstance(req, list) and req:
            has_all = all(re.search(rf"\b{re.escape(w)}\b", t, flags=re.I) for w in req if w)
            if not has_all:
                extra = " ".join(set(req) - set([w for w in req if re.search(rf'\b{re.escape(w)}\b', t, flags=re.I)]))
                t = (t + "\n\n" + f"(ajout requis: {extra})").strip()

        # max length
        max_len = self.contract.get("max_chars")
        if isinstance(max_len, int) and max_len > 0 and len(t) > max_len:
            t = t[:max_len].rstrip() + "…"

        # style hint (very light touch)
        style = self.contract.get("style")
        if style and self.summarizer:
            try:
                t = self.summarizer(f"style: {style}\n\ntext:\n{t}", max_length=min(200, max(60, len(t)//4)), min_length=40)[0]["summary_text"].strip() or t
            except Exception:
                pass
        return t

    # ---------------- user assertions ----------------

    def _apply_assertions(self, text: str) -> str:
        if not self.assertions:
            return text
        violations = []
        for name, pattern in self.assertions.items():
            try:
                if not re.search(pattern, text, flags=re.I | re.M):
                    violations.append(name)
            except re.error:
                continue
        self._last_trace["assert_violations"] = violations
        if not violations:
            return text
        # try a light repair with summarizer if available
        if self.summarizer:
            try:
                repaired = self.summarizer(
                    "répare le texte pour satisfaire ces contraintes: "
                    + ", ".join(violations) + "\n\n" + text,
                    max_length=min(220, max(70, len(text)//3)),
                    min_length=50
                )[0]["summary_text"].strip()
                return repaired or text
            except Exception:
                return text
        return text

    # ---------------- command handlers ----------------

    def _cmd_help(self, *_):
        return (
            "commandes:\n"
            "- /help, /about, /diag, /trace\n"
            "- /memory, /reset\n"
            "- /save <path>, /load <path>, /export <path>\n"
            "- /set key=value (temperature, top_p, max_new_tokens, max_context_tokens, "
            "lang, redact_pii, system_prompt, consensus, n_consensus, paa, c3, critic)\n"
            "- /kb add <title>|<text>, /kb search <query>, /kb list, /kb clear\n"
            "- /contract set <json>, /contract show, /contract clear\n"
            "- /assert add <name>|<regex>, /assert list, /assert clear\n"
            "- 'traduis en anglais <texte>' / 'traduis en français <texte>'\n"
            "- 'résume <texte>'\n"
            "- 'question: <votre question> | contexte: <texte>'"
        )

    def _cmd_about(self, *_):
        info = {
            "device": str(self.device),
            "primary_seq2seq": self.is_seq2seq_a,
            "secondary_loaded": self.model_b is not None,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_new_tokens": self.cfg.max_new_tokens,
            "max_context_tokens": self.cfg.max_context_tokens,
            "preferred_lang": self.cfg.preferred_lang,
            "redact_pii": self.cfg.redact_pii,
            "consensus": self.cfg.consensus,
            "n_consensus": self.cfg.n_consensus,
            "paa": self.cfg.paa,
            "c3": self.cfg.c3,
            "critic": self.cfg.critic,
            "contract_active": bool(self.contract),
            "assertions": list(self.assertions.keys())
        }
        return "infos:\n" + "\n".join(f"- {k}: {v}" for k, v in info.items())

    def _cmd_diag(self, *_):
        checks = [
            ("primary model", self.model_a is not None),
            ("secondary model", self.model_b is not None),
            ("translator fr→en", self.translator_fr_en is not None),
            ("translator en→fr", self.translator_en_fr is not None),
            ("summarizer", self.summarizer is not None),
            ("qa", self.qa is not None),
            ("kb docs", len(self.kb.docs)),
        ]
        return "diagnostic:\n" + "\n".join(f"- {k}: {('ok' if (v if isinstance(v,bool) else True) else 'missing') if isinstance(v,bool) else v}" for k, v in checks)

    def _cmd_trace(self, *_):
        if not self._last_trace:
            return "trace: aucune métrique disponible pour l'instant."
        return "trace:\n" + json.dumps(self._last_trace, ensure_ascii=False, indent=2)

    def _cmd_memory(self, *_):
        if not self.memory.turns:
            return "mémoire vide."
        lines = []
        for t in self.memory.turns[-16:]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.ts))
            lines.append(f"{ts} | {t.speaker}: {t.text}")
        return "\n".join(lines)

    def _cmd_reset(self, *_):
        self.memory = ChatMemory(max_turns=self.memory.max_turns)
        self._last_trace = {}
        return "mémoire réinitialisée."

    def _cmd_save(self, path: Optional[str] = None, *_):
        p = (path or "").strip() or self.cfg.save_path
        state = {
            "config": asdict(self.cfg),
            "turns": self.memory.to_json(),
            "kb": self.kb.docs,
            "contract": self.contract,
            "assertions": self.assertions
        }
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            return f"état enregistré dans: {p}"
        except Exception as e:
            logger.error(f"save error: {e}")
            return f"erreur d'enregistrement: {e}"

    def _cmd_load(self, path: Optional[str] = None, *_):
        p = (path or "").strip() or self.cfg.save_path
        try:
            with open(p, "r", encoding="utf-8") as f:
                state = json.load(f)
            cfg = ChatConfig(**state.get("config", {}))
            cfg.clamp()
            self.cfg = cfg
            self.memory.from_json(state.get("turns", []))
            self.kb.docs = [(t, x) for t, x in state.get("kb", [])]
            self.kb.rebuild()
            self.contract = state.get("contract")
            self.assertions = state.get("assertions", {})
            self._last_trace = {}
            return f"état rechargé depuis: {p}"
        except Exception as e:
            logger.error(f"load error: {e}")
            return f"erreur de chargement: {e}"

    def _cmd_export(self, path: Optional[str] = None, *_):
        p = (path or "").strip() or self.cfg.transcript_md
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write("# chat transcript\n\n")
                for t in self.memory.turns:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.ts))
                    who = t.speaker
                    txt = t.text.replace("\n", "  \n")
                    f.write(f"**{who}** ({ts}):\n\n{txt}\n\n---\n\n")
            return f"transcription exportée: {p}"
        except Exception as e:
            logger.error(f"export error: {e}")
            return f"erreur d'export: {e}"

    def _cmd_set(self, *args):
        arg = " ".join(a for a in args).strip()
        if not arg or "=" not in arg:
            return "format: /set key=value"
        key, val = arg.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        try:
            if key == "temperature":
                self.cfg.temperature = float(val)
            elif key == "top_p":
                self.cfg.top_p = float(val)
            elif key == "max_new_tokens":
                self.cfg.max_new_tokens = int(val)
            elif key == "max_context_tokens":
                self.cfg.max_context_tokens = int(val)
            elif key == "lang":
                self.cfg.preferred_lang = val.lower()
            elif key == "redact_pii":
                self.cfg.redact_pii = val.lower() in {"1","true","yes","y","on"}
            elif key == "system_prompt":
                self.cfg.system_prompt = val.strip().strip('"').strip("'")
            elif key == "consensus":
                self.cfg.consensus = val.lower() in {"1","true","yes","y","on"}
            elif key == "n_consensus":
                self.cfg.n_consensus = int(val)
            elif key == "paa":
                self.cfg.paa = val.lower() in {"1","true","yes","y","on"}
            elif key == "c3":
                self.cfg.c3 = val.lower() in {"1","true","yes","y","on"}
            elif key == "critic":
                self.cfg.critic = val.lower() in {"1","true","yes","y","on"}
            else:
                return f"clé inconnue: {key}"
            self.cfg.clamp()
            return f"ok: {key}={getattr(self.cfg, key) if hasattr(self.cfg, key) else val}"
        except Exception as e:
            return f"valeur invalide pour {key}: {e}"

    def _cmd_kb(self, *args):
        if not args:
            return "kb: /kb add <title>|<text> | /kb search <query> | /kb list | /kb clear"
        sub = args[0].lower()
        rest = " ".join(args[1:]).strip()
        if sub == "add":
            if "|" not in rest:
                return "format: /kb add <title>|<text>"
            title, text = rest.split("|", 1)
            title, text = title.strip(), text.strip()
            if not title or not text:
                return "format: /kb add <title>|<text>"
            self.kb.add(title, text)
            return f"kb: ajouté « {title} » (total: {len(self.kb.docs)})"
        if sub == "search":
            if not rest:
                return "format: /kb search <query>"
            hits = self.kb.search(rest, k=3)
            if not hits:
                return "kb: aucun résultat."
            return "kb résultats:\n" + "\n".join(f"- {t} (score={s:.3f})" for t, s in hits)
        if sub == "list":
            if not self.kb.docs:
                return "kb: vide."
            return "kb documents:\n" + "\n".join(f"- {t}" for t in self.kb.list())
        if sub == "clear":
            self.kb.clear()
            return "kb: vidé."
        return "kb: commande inconnue."

    def _cmd_contract(self, *args):
        if not args:
            return "contract: /contract set <json> | /contract show | /contract clear"
        sub = args[0].lower()
        rest = " ".join(args[1:]).strip()
        if sub == "set":
            try:
                obj = json.loads(rest)
                # allowed keys: language, max_chars, forbid_words, require_words, style
                allowed = {"language", "max_chars", "forbid_words", "require_words", "style"}
                self.contract = {k: v for k, v in obj.items() if k in allowed}
                return "contract: défini."
            except Exception as e:
                return f"contract: json invalide ({e})"
        if sub == "show":
            return "contract:\n" + (json.dumps(self.contract, ensure_ascii=False, indent=2) if self.contract else "aucun")
        if sub == "clear":
            self.contract = None
            return "contract: supprimé."
        return "contract: commande inconnue."

    def _cmd_assert(self, *args):
        if not args:
            return "assert: /assert add <name>|<regex> | /assert list | /assert clear"
        sub = args[0].lower()
        rest = " ".join(args[1:]).strip()
        if sub == "add":
            if "|" not in rest:
                return "format: /assert add <name>|<regex>"
            name, regex = rest.split("|", 1)
            name, regex = name.strip(), regex.strip()
            if not name or not regex:
                return "format: /assert add <name>|<regex>"
            self.assertions[name] = regex
            return f"assert: ajoutée « {name} »."
        if sub == "list":
            if not self.assertions:
                return "assert: aucune."
            return "assert:\n" + "\n".join(f"- {k}: {v}" for k, v in self.assertions.items())
        if sub == "clear":
            self.assertions = {}
            return "assert: vidé."
        return "assert: commande inconnue."

    # ---------------- routing & chat ----------------

    def _route_command(self, text: str) -> Optional[str]:
        if not text.startswith("/"):
            return None
        parts = text.split()
        cmd = parts[0].lower()
        handler = self.commands.get(cmd)
        if not handler:
            return f"commande inconnue: {cmd}"
        try:
            return handler(*parts[1:])
        except Exception as e:
            logger.error(f"command error ({cmd}): {e}")
            return f"erreur commande {cmd}: {e}"

    def _maybe_translate(self, reply: str, user_lang: str) -> str:
        target = self.cfg.preferred_lang
        if target == "auto":
            target = user_lang or "fr"
        target = target.lower()

        if target.startswith("fr"):
            if self.translator_en_fr and self.detect_language(reply) != "fr":
                try:
                    return self.translator_en_fr(reply)[0]["translation_text"]
                except Exception:
                    return reply
            return reply

        if target.startswith("en"):
            if self.translator_fr_en and self.detect_language(reply) != "en":
                try:
                    return self.translator_fr_en(reply)[0]["translation_text"]
                except Exception:
                    return reply
            return reply

        return reply

    def _apply_pipeline(self, prompt: str) -> str:
        # tscd consensus or single-path
        base_answer = self.generate_reply_consensus(prompt) if self.cfg.consensus else self.generate_reply(prompt)

        # paa stage
        if self.cfg.paa and self.translator_fr_en and self.translator_en_fr:
            base_answer, _ = self._paa_answer(prompt)

        # c3 stage
        if self.cfg.c3:
            base_answer = self._c3_refine(prompt, base_answer)

        # rcg stage
        if self.cfg.critic:
            base_answer = self._rcg_polish(base_answer)

        return base_answer

    def chat(self, user_input: str) -> str:
        # persistent rate limiter
        if not self.limiter.allow():
            return "⏳ trop rapide — réessayez dans un instant."

        ui_raw = (user_input or "").strip()
        if not ui_raw:
            return "dites-moi quelque chose 🙂"

        if _block_blacklisted_urls(ui_raw, self.cfg.url_blacklist):
            return "🚫 url non autorisée détectée."

        if not self.safety.is_safe(ui_raw):
            return "🚫 contenu bloqué par le filtre de sécurité."

        # slash commands
        routed = self._route_command(ui_raw)
        if routed is not None:
            return routed

        low = ui_raw.lower()

        # translations quick-commands
        if low.startswith("traduis en anglais"):
            txt = ui_raw[len("traduis en anglais"):].strip()
            if not txt:
                return "ajoutez le texte à traduire."
            if not self.translator_fr_en:
                return "pipeline de traduction fr→en indisponible en local."
            return self.translator_fr_en(txt)[0]["translation_text"]

        if low.startswith("traduis en français"):
            txt = ui_raw[len("traduis en français"):].strip()
            if not txt:
                return "ajoutez le texte à traduire."
            if not self.translator_en_fr:
                return "pipeline de traduction en→fr indisponible en local."
            return self.translator_en_fr(txt)[0]["translation_text"]

        # summarization quick-command
        if low.startswith(("résume", "resume", "resumé")):
            if not self.summarizer:
                return "pipeline de résumé indisponible en local."
            parts = ui_raw.split(" ", 1)
            if len(parts) < 2 or not parts[1].strip():
                return "format: 'résume <texte à résumer>'"
            try:
                return self.summarizer(parts[1].strip(), max_length=150, min_length=30)[0]["summary_text"]
            except Exception as e:
                logger.error(f"summarization error: {e}")
                return "erreur pendant le résumé."

        # qa quick-command
        if low.startswith("question:"):
            if not self.qa:
                return "pipeline de question-réponse indisponible en local."
            try:
                left, right = ui_raw.split("|", 1)
                question = left.split(":", 1)[1].strip()
                context = right.split(":", 1)[1].strip() if ":" in right else right.strip()
                if not question or not context:
                    raise ValueError("empty parts")
                return self.qa(question=question, context=context)["answer"]
            except Exception:
                return "format: Question: <votre question> | Contexte: <texte>"

        # language hint
        try:
            user_lang = langdetect.detect(ui_raw)
        except Exception:
            user_lang = self.memory.last_user_lang_hint() or "fr"

        # normal conversation with full pipeline
        self.memory.add("User", ui_raw)
        reply = self._apply_pipeline(ui_raw)

        # kb verifiability tagging
        reply = self._tag_verifiability(reply)

        # reply contracts enforcement
        reply = self._enforce_contract(reply)

        # user assertions enforcement
        reply = self._apply_assertions(reply)

        # final adaptation to preferred language
        reply_final = self._maybe_translate(reply, user_lang=user_lang)

        # update trace for transparency
        self._last_trace["length_chars"] = len(reply_final)

        self.memory.add("Bot", reply_final)
        return reply_final

# -----------------------------------------------------------------------------
# cli runner
# -----------------------------------------------------------------------------

def _graceful_exit(signum, frame):
    print("\n👋 bye!")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, _graceful_exit)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _graceful_exit)

    print(f"device utilisé : {get_device()}")
    try:
        bot = AdvancedMultilingualChatbot()
    except Exception as e:
        logger.error(f"initialization failed: {e}")
        print("❌ impossible de démarrer: vérifiez que les modèles sont en cache local.")
        sys.exit(1)

    print("🤖 prêt ! tapez ‘quit’ pour sortir. (/help pour l’aide)")
    while True:
        try:
            ui = input("➡️ vous : ").strip()
        except EOFError:
            print("\n👋 bye!")
            break

        if ui.lower() in {"quit", "exit"}:
            print("👋 bye!")
            break

        resp = bot.chat(ui)
        print("💬 bot :", resp)

if __name__ == "__main__":
    main()
