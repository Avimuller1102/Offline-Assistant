# echoshield_offline.py
# a fully offline, multilingual chatbot with self-consensus + counterfactual checks
# + proof-carrying answers (pca) against a local kb.
#
# goals:
# - runs with no network calls (transformers local_files_only)
# - usable even when some optional pipelines are missing (degrades gracefully)
# - exports a "proof pack" json for auditing: evidence + entailment + kb hash
#
# dependencies:
# - python 3.9+
# - torch
# - transformers
# - langdetect
#
# optional (only if available locally):
# - translation models for back-translation (paa)
# - summarization model for fusion/polish
# - nli model (mnli) for entailment-based verification
#
# note: "perfect" software does not exist; this is engineered to be robust, debuggable, and auditable.

import os
import re
import sys
import json
import time
import math
import uuid
import signal
import hashlib
import logging
import unicodedata
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
import langdetect
from functools import lru_cache

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------

LOG_LEVEL = os.getenv("ECHOSHIELD_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("echoshield")
logging.getLogger("transformers").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# text utils
# -----------------------------------------------------------------------------

def now_ts() -> float:
    return time.time()

def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())

def strip_control_and_invisibles(text: str) -> str:
    # remove control chars + suspicious invisible unicode (prompt-injection vector)
    if not text:
        return ""
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        # keep normal whitespace and printable chars
        if ch in "\n\t\r":
            cleaned.append(ch)
            continue
        # drop control (c*) and format (cf) chars except basic spaces
        if cat.startswith("C"):
            continue
        cleaned.append(ch)
    return "".join(cleaned)

def sentence_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[a-zàâäéèêëîïôöùûüç0-9]+", (text or "").lower(), flags=re.I)

def repetition_ratio(text: str) -> float:
    toks = tokenize_simple(text)
    if not toks:
        return 0.0
    return max(0.0, 1.0 - (len(set(toks)) / max(1, len(toks))))

def ngrams(tokens: List[str], n: int) -> set:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def overlap_score(a: str, b: str) -> float:
    ta, tb = tokenize_simple(a), tokenize_simple(b)
    if not ta or not tb:
        return 0.0
    score = 0.0
    for n, w in [(1, 0.25), (2, 0.45), (3, 0.65)]:
        na, nb = ngrams(ta, n), ngrams(tb, n)
        inter = len(na & nb)
        union = len(na | nb) or 1
        score += w * (inter / union)
    rep_pen = min(0.25, max(0.0, (repetition_ratio(a) + repetition_ratio(b)) / 2.0))
    return max(0.0, score - rep_pen)

def pii_redact(text: str) -> str:
    # minimal pii redaction (demo-level)
    t = text or ""
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[redacted-email]", t)
    t = re.sub(r"\b(\+?\d[\d\-\s]{6,}\d)\b", "[redacted-phone]", t)
    t = re.sub(r"(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9\-_.]{8,}['\"]?",
               r"\1: [redacted-secret]", t, flags=re.I)
    return t

# -----------------------------------------------------------------------------
# prompt injection shield (lightweight, offline)
# -----------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"\bignore (all|any|previous) (instructions|rules)\b",
    r"\boverride (system|developer) (message|prompt)\b",
    r"\bdisregard (?:safety|policy|guardrails)\b",
    r"\breveal (?:system|developer) prompt\b",
    r"\byou are now (?:system|developer)\b",
    r"\bact as (?:system|developer)\b",
]

def strip_injection(text: str) -> str:
    t = text or ""
    for pat in INJECTION_PATTERNS:
        t = re.sub(pat, "[removed]", t, flags=re.I)
    return t

# -----------------------------------------------------------------------------
# device utils
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_torch_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

torch.set_grad_enabled(False)
set_torch_seed(42)

# -----------------------------------------------------------------------------
# rate limiting
# -----------------------------------------------------------------------------

class RateLimiter:
    # token bucket + flood guard
    def __init__(self, rpm: int = 60, min_gap_s: float = 0.25):
        self.capacity = max(1, int(rpm))
        self.tokens = float(self.capacity)
        self.refill_rate = float(self.capacity) / 60.0
        self.last = now_ts()
        self.min_gap_s = float(min_gap_s)
        self.last_call = 0.0

    def allow(self) -> bool:
        now = now_ts()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        if (now - self.last_call) < self.min_gap_s:
            return False
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.last_call = now
            return True
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
    def __init__(self, max_turns: int = 10):
        self.max_turns = int(max_turns)
        self.turns: List[ChatTurn] = []

    def add(self, speaker: str, text: str) -> None:
        self.turns.append(ChatTurn(speaker=speaker, text=normalize_ws(text), ts=now_ts()))
        if len(self.turns) > 2 * self.max_turns:
            self.turns = self.turns[-2 * self.max_turns :]

    def context(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)

    def to_json(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self.turns]

    def from_json(self, data: List[Dict[str, Any]]) -> None:
        self.turns = [ChatTurn(**d) for d in (data or [])][-2 * self.max_turns :]

    def last_user_lang(self) -> Optional[str]:
        for t in reversed(self.turns):
            if t.speaker.lower() == "user":
                try:
                    return langdetect.detect(t.text)
                except Exception:
                    return None
        return None

# -----------------------------------------------------------------------------
# kb: bm25 (no external deps), with content hash ledger
# -----------------------------------------------------------------------------

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class BM25KB:
    # small bm25 kb with json persistence
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []  # {id, title, text}
        self.df: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self._built = False

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return tokenize_simple(text)

    def _rebuild(self) -> None:
        self.df = {}
        total_len = 0
        for d in self.docs:
            toks = set(self._tokens(d["text"]))
            total_len += len(self._tokens(d["text"]))
            for t in toks:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (total_len / max(1, len(self.docs))) if self.docs else 0.0
        self._built = True

    def ledger_hash(self) -> str:
        # stable hash of kb contents (for audit)
        payload = json.dumps(
            [{"title": d["title"], "text": d["text"]} for d in self.docs],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return sha256_hex(payload)

    def add(self, title: str, text: str) -> str:
        doc_id = str(uuid.uuid4())
        self.docs.append({"id": doc_id, "title": title.strip(), "text": text.strip()})
        self._rebuild()
        return doc_id

    def clear(self) -> None:
        self.docs = []
        self._rebuild()

    def list_titles(self) -> List[str]:
        return [d["title"] for d in self.docs]

    def save(self, path: str) -> None:
        obj = {"docs": self.docs}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.docs = obj.get("docs", [])
        self._rebuild()

    def search(self, query: str, k: int = 5, k1: float = 1.2, b: float = 0.75) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        if not self._built:
            self._rebuild()

        q_toks = self._tokens(query)
        q_counts: Dict[str, int] = {}
        for t in q_toks:
            q_counts[t] = q_counts.get(t, 0) + 1

        N = len(self.docs)
        results = []
        for d in self.docs:
            d_toks = self._tokens(d["text"])
            dl = len(d_toks)
            tf: Dict[str, int] = {}
            for t in d_toks:
                tf[t] = tf.get(t, 0) + 1

            score = 0.0
            for term in q_counts.keys():
                if term not in tf:
                    continue
                df = self.df.get(term, 0)
                # idf with bm25+ style smoothing
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                denom = tf[term] + k1 * (1.0 - b + b * (dl / (self.avgdl or 1.0)))
                score += idf * (tf[term] * (k1 + 1.0) / (denom or 1.0))

            if score > 0:
                results.append({"id": d["id"], "title": d["title"], "text": d["text"], "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[: max(1, int(k))]

# -----------------------------------------------------------------------------
# safety
# -----------------------------------------------------------------------------

class SafetyFilter:
    # offline safety: regex flags + optional mnli zero-shot
    def __init__(self, device: torch.device, threshold: float = 0.60):
        self.threshold = float(threshold)
        self.enabled = True
        self.labels = ["hate", "harassment", "self-harm", "sexual", "violence"]

        self._regex_flags = [
            re.compile(r"\bhow to make (a )?bomb\b", re.I),
            re.compile(r"\bsuicide\b", re.I),
            re.compile(r"\bkill\s+(myself|yourself|himself|herself|them|me)\b", re.I),
        ]

        self.moderator = None
        try:
            self.moderator = pipeline(
                task="zero-shot-classification",
                model="facebook/bart-large-mnli",
                tokenizer="facebook/bart-large-mnli",
                device=0 if device.type == "cuda" else -1,
                local_files_only=True,
            )
            logger.info("safety: mnli loaded.")
        except Exception as e:
            logger.warning(f"safety: mnli not available locally ({e}); regex-only mode.")
            self.moderator = None

    def is_safe(self, text: str) -> bool:
        t = normalize_ws(text)
        for pat in self._regex_flags:
            if pat.search(t):
                return False
        if not self.moderator:
            return True
        try:
            res = self.moderator(t, candidate_labels=self.labels)
            scores = {lbl.lower(): sc for lbl, sc in zip(res["labels"], res["scores"])}
            return not any(scores.get(lbl, 0.0) >= self.threshold for lbl in self.labels)
        except Exception:
            # fail-open here to avoid false blocks on pipeline errors
            return True

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------

@dataclass
class EchoConfig:
    # generation
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 220
    max_context_tokens: int = 1536
    do_sample: bool = True

    # consensus
    consensus: bool = True
    n_consensus: int = 5

    # robustness checks
    paa: bool = True
    c3: bool = True
    critic: bool = True

    # verification
    verify_with_nli: bool = True
    nli_supported_threshold: float = 0.62
    nli_contradiction_threshold: float = 0.62
    bm25_min_score_for_claim: float = 1.0

    # safety + hygiene
    redact_pii: bool = True
    safety_threshold: float = 0.60
    rpm_limit: int = 60
    flood_seconds: float = 0.35

    # output control
    preferred_lang: str = "auto"  # auto/fr/en
    system_prompt: str = (
        "you are a helpful, concise, and safe assistant. "
        "when unsure, say so. do not invent facts. prefer the local kb as source of truth."
    )

    # persistence
    state_path: str = "echoshield_state.json"
    kb_path: str = "echoshield_kb.json"
    transcript_path: str = "echoshield_transcript.md"
    proofpack_path: str = "echoshield_proofpack.json"

    def clamp(self) -> None:
        self.temperature = float(min(max(self.temperature, 0.05), 1.5))
        self.top_p = float(min(max(self.top_p, 0.1), 1.0))
        self.max_new_tokens = int(min(max(self.max_new_tokens, 16), 512))
        self.max_context_tokens = int(min(max(self.max_context_tokens, 256), 4096))
        self.rpm_limit = int(min(max(self.rpm_limit, 1), 600))
        self.flood_seconds = float(min(max(self.flood_seconds, 0.05), 5.0))
        self.n_consensus = int(min(max(self.n_consensus, 1), 9))

# -----------------------------------------------------------------------------
# model loader (offline)
# -----------------------------------------------------------------------------

class OfflineModels:
    def __init__(self, device: torch.device):
        self.device = device

        # main conversational candidates (you can override via env)
        self.chat_candidates = [
            os.getenv("ECHOSHIELD_MODEL_A", "facebook/blenderbot-400M-distill"),
            os.getenv("ECHOSHIELD_MODEL_B", "facebook/mbart-large-50-many-to-many-mmt"),
        ]

        self.tokenizer_a = None
        self.model_a = None
        self.is_seq2seq_a = True

        self.tokenizer_b = None
        self.model_b = None
        self.is_seq2seq_b = True

        # optional pipelines
        self.trans_fr_en = None
        self.trans_en_fr = None
        self.summarizer = None
        self.nli = None

        self._load_chat_models()
        self._load_optional_pipelines()

    def _load_one(self, name: str):
        cfg = AutoConfig.from_pretrained(name, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        is_s2s = bool(getattr(cfg, "is_encoder_decoder", False))
        if is_s2s:
            mdl = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True)
        else:
            mdl = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
        mdl.to(self.device).eval()
        mdl.config.pad_token_id = tok.pad_token_id
        return tok, mdl, is_s2s

    def _load_chat_models(self):
        last_err = None
        for name in self.chat_candidates:
            try:
                tok, mdl, is_s2s = self._load_one(name)
                if not self.tokenizer_a:
                    self.tokenizer_a, self.model_a, self.is_seq2seq_a = tok, mdl, is_s2s
                    logger.info(f"chat primary loaded: {name} (seq2seq={is_s2s})")
                else:
                    self.tokenizer_b, self.model_b, self.is_seq2seq_b = tok, mdl, is_s2s
                    logger.info(f"chat secondary loaded: {name} (seq2seq={is_s2s})")
                    break
            except Exception as e:
                last_err = e
                logger.warning(f"chat model not available locally: {name}: {e}")
        if not self.model_a:
            raise RuntimeError(f"no local chat model available. last error: {last_err}")

    def _pipe(self, task: str, model_name: str):
        try:
            pl = pipeline(
                task=task,
                model=model_name,
                device=0 if self.device.type == "cuda" else -1,
                local_files_only=True,
            )
            logger.info(f"pipeline loaded: {task} ({model_name})")
            return pl
        except Exception as e:
            logger.warning(f"pipeline missing locally: {task} ({model_name}): {e}")
            return None

    def _load_optional_pipelines(self):
        # translation (paa)
        self.trans_fr_en = self._pipe("translation_fr_to_en", "Helsinki-NLP/opus-mt-fr-en")
        self.trans_en_fr = self._pipe("translation_en_to_fr", "Helsinki-NLP/opus-mt-en-fr")

        # summarizer (fusion/polish)
        self.summarizer = self._pipe("summarization", "facebook/bart-large-cnn")

        # nli entailment verifier
        # we reuse mnli in a direct entailment setting with labels.
        # the zero-shot pipeline is not ideal for entailment; so we use text-classification if available.
        try:
            self.nli = pipeline(
                task="text-classification",
                model="facebook/bart-large-mnli",
                tokenizer="facebook/bart-large-mnli",
                device=0 if self.device.type == "cuda" else -1,
                local_files_only=True,
                return_all_scores=True,
            )
            logger.info("nli verifier loaded: facebook/bart-large-mnli")
        except Exception as e:
            logger.warning(f"nli verifier missing locally: {e}")
            self.nli = None

# -----------------------------------------------------------------------------
# core: proof-carrying answer verification
# -----------------------------------------------------------------------------

def mnli_entailment_probs(nli_pipe, premise: str, hypothesis: str) -> Optional[Dict[str, float]]:
    # returns probabilities for entailment/neutral/contradiction if available
    # bart-mnli labels are typically: CONTRADICTION, NEUTRAL, ENTAILMENT
    if not nli_pipe:
        return None
    try:
        inp = premise + " </s></s> " + hypothesis
        out = nli_pipe(inp)
        # out: [[{'label': 'CONTRADICTION', 'score': ...}, ...]]
        scores = {d["label"].upper(): float(d["score"]) for d in out[0]}
        return {
            "ENTAILMENT": scores.get("ENTAILMENT", 0.0),
            "NEUTRAL": scores.get("NEUTRAL", 0.0),
            "CONTRADICTION": scores.get("CONTRADICTION", 0.0),
        }
    except Exception:
        return None

@dataclass
class EvidenceItem:
    doc_id: str
    title: str
    bm25_score: float
    entailment: Optional[Dict[str, float]] = None

@dataclass
class ClaimVerdict:
    sentence: str
    verdict: str  # supported/contradicted/unverified
    evidence: List[EvidenceItem]

# -----------------------------------------------------------------------------
# contracts / assertions
# -----------------------------------------------------------------------------

@dataclass
class ReplyContract:
    language: Optional[str] = None      # "fr" or "en"
    max_chars: Optional[int] = None
    forbid_words: Optional[List[str]] = None
    require_words: Optional[List[str]] = None
    style: Optional[str] = None         # just a hint

class ContractEngine:
    def __init__(self):
        self.contract: Optional[ReplyContract] = None
        self.assertions: Dict[str, str] = {}  # name->regex

    def set_contract(self, obj: Dict[str, Any]) -> None:
        allowed = {"language", "max_chars", "forbid_words", "require_words", "style"}
        clean = {k: obj.get(k) for k in allowed if k in obj}
        self.contract = ReplyContract(**clean)

    def clear_contract(self) -> None:
        self.contract = None

    def add_assertion(self, name: str, regex: str) -> None:
        self.assertions[name] = regex

    def clear_assertions(self) -> None:
        self.assertions = {}

    def enforce(self, text: str) -> str:
        if not self.contract:
            return text
        t = text

        # forbid words
        if self.contract.forbid_words:
            for w in self.contract.forbid_words:
                if w:
                    t = re.sub(rf"\b{re.escape(w)}\b", "[removed]", t, flags=re.I)

        # require words
        if self.contract.require_words:
            missing = []
            for w in self.contract.require_words:
                if w and not re.search(rf"\b{re.escape(w)}\b", t, flags=re.I):
                    missing.append(w)
            if missing:
                t = (t + "\n\n" + "(ajout requis: " + ", ".join(missing) + ")").strip()

        # max length
        if isinstance(self.contract.max_chars, int) and self.contract.max_chars > 0:
            if len(t) > self.contract.max_chars:
                t = t[: self.contract.max_chars].rstrip() + "…"

        return t

    def check_assertions(self, text: str) -> List[str]:
        violations = []
        for name, pattern in self.assertions.items():
            try:
                if not re.search(pattern, text, flags=re.I | re.M):
                    violations.append(name)
            except re.error:
                violations.append(name)
        return violations

# -----------------------------------------------------------------------------
# the chatbot
# -----------------------------------------------------------------------------

class EchoShield:
    def __init__(self, cfg: Optional[EchoConfig] = None):
        self.cfg = cfg or EchoConfig()
        self.cfg.clamp()

        self.device = get_device()
        logger.info(f"device: {self.device}")

        self.models = OfflineModels(self.device)
        self.safety = SafetyFilter(self.device, threshold=self.cfg.safety_threshold)
        self.kb = BM25KB()
        self.memory = ChatMemory(max_turns=10)
        self.limiter = RateLimiter(self.cfg.rpm_limit, self.cfg.flood_seconds)
        self.contracts = ContractEngine()

        self._last_trace: Dict[str, Any] = {}
        self._last_proofpack: Optional[Dict[str, Any]] = None

        # load kb if exists
        if os.path.exists(self.cfg.kb_path):
            try:
                self.kb.load(self.cfg.kb_path)
                logger.info(f"kb loaded from {self.cfg.kb_path} ({len(self.kb.docs)} docs)")
            except Exception as e:
                logger.warning(f"kb load failed: {e}")

    @lru_cache(maxsize=1024)
    def detect_lang(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except Exception:
            return "en"

    def _build_prompt(self, user_text: str) -> str:
        sys_p = f"System: {self.cfg.system_prompt}"
        ctx = self.memory.context()
        if ctx:
            return f"{sys_p}\n{ctx}\nUser: {user_text}\nBot:"
        return f"{sys_p}\nUser: {user_text}\nBot:"

    def _truncate(self, input_ids: torch.Tensor, reserve: int, model) -> torch.Tensor:
        model_max = int(getattr(model.config, "max_position_embeddings", 2048))
        model_max = max(model_max, 512)
        max_input = max(min(self.cfg.max_context_tokens, model_max - reserve - 8), 16)
        if input_ids.shape[-1] > max_input:
            return input_ids[:, -max_input:]
        return input_ids

    def _gen_params(self, tokenizer) -> Dict[str, Any]:
        return dict(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=bool(self.cfg.do_sample),
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    def _generate_once(self, model, tokenizer, is_seq2seq: bool, user_text: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            set_torch_seed(seed)

        t = strip_control_and_invisibles(user_text)
        t = normalize_ws(t)
        t = strip_injection(t)
        if self.cfg.redact_pii:
            t = pii_redact(t)

        prompt = self._build_prompt(t)
        enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = enc["input_ids"].to(self.device)
        input_ids = self._truncate(input_ids, reserve=self.cfg.max_new_tokens, model=model)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        params = self._gen_params(tokenizer)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **params,
            )

        if is_seq2seq:
            decoded = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        else:
            cont = out[0][input_ids.shape[-1] :]
            decoded = tokenizer.decode(cont, skip_special_tokens=True).strip()

        return decoded or "je n'ai pas bien compris, pouvez-vous reformuler ?"

    def _consensus(self, user_text: str) -> Tuple[str, Dict[str, Any]]:
        n = self.cfg.n_consensus if self.cfg.consensus else 1
        n = max(1, int(n))

        seeds = [(int(now_ts()) % 100000) + i * 97 for i in range(n)]
        candidates = []
        meta = {"seeds": seeds, "candidates": []}

        for i, sd in enumerate(seeds):
            use_b = (self.models.model_b is not None) and (i % 2 == 1)
            model = self.models.model_b if use_b else self.models.model_a
            tok = self.models.tokenizer_b if use_b else self.models.tokenizer_a
            is_s2s = self.models.is_seq2seq_b if use_b else self.models.is_seq2seq_a

            try:
                ans = self._generate_once(model, tok, is_s2s, user_text, seed=sd)
            except Exception as e:
                logger.warning(f"generation failed (seed={sd}): {e}")
                ans = self._generate_once(self.models.model_a, self.models.tokenizer_a, self.models.is_seq2seq_a, user_text)

            candidates.append(ans)
            meta["candidates"].append({"seed": sd, "use_secondary": bool(use_b), "text": ans})

        if len(candidates) == 1:
            return candidates[0], meta

        # pick by agreement to others + low repetition
        scores = []
        for i in range(len(candidates)):
            agree = 0.0
            for j in range(len(candidates)):
                if i == j:
                    continue
                agree += overlap_score(candidates[i], candidates[j])
            agree /= max(1, len(candidates) - 1)
            rep = repetition_ratio(candidates[i])
            scores.append(agree - 0.25 * rep)

        best_i = max(range(len(candidates)), key=lambda i: scores[i])
        best = candidates[best_i]
        meta["consensus_scores"] = scores
        meta["consensus_best_index"] = best_i

        # optional fusion: if top two agree enough
        if self.models.summarizer and len(candidates) >= 2:
            top2 = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:2]
            a, b = candidates[top2[0]], candidates[top2[1]]
            if overlap_score(a, b) >= 0.35:
                try:
                    fused = self.models.summarizer(a + "\n\n" + b, max_length=180, min_length=50)[0]["summary_text"].strip()
                    if fused:
                        meta["fused_from"] = top2
                        best = fused
                except Exception:
                    pass

        return best, meta

    # paa: paraphrase-and-agree via back-translation
    def _back_translate(self, text: str) -> str:
        lang = self.detect_lang(text)
        try:
            if lang.startswith("fr") and self.models.trans_fr_en and self.models.trans_en_fr:
                en = self.models.trans_fr_en(text)[0]["translation_text"]
                fr = self.models.trans_en_fr(en)[0]["translation_text"]
                return fr
            if lang.startswith("en") and self.models.trans_en_fr and self.models.trans_fr_en:
                fr = self.models.trans_en_fr(text)[0]["translation_text"]
                en = self.models.trans_fr_en(fr)[0]["translation_text"]
                return en
        except Exception:
            return text
        return text

    def _paa_stage(self, user_text: str) -> Tuple[str, Dict[str, Any]]:
        if not self.cfg.paa:
            ans, meta = self._consensus(user_text)
            return ans, {"paa": False, **meta}

        # if translators missing, skip
        if not (self.models.trans_fr_en and self.models.trans_en_fr):
            ans, meta = self._consensus(user_text)
            return ans, {"paa": False, "paa_reason": "translation pipelines not available", **meta}

        p1 = self._back_translate(user_text)
        p2 = self._back_translate(p1 + " ")

        a0, m0 = self._consensus(user_text)
        a1, m1 = self._consensus(p1)
        a2, m2 = self._consensus(p2)

        best = a0
        # pick by mutual overlap among answers
        cand = [a0, a1, a2]
        agree_scores = []
        for i in range(3):
            s = 0.0
            for j in range(3):
                if i == j:
                    continue
                s += overlap_score(cand[i], cand[j])
            agree_scores.append(s / 2.0)
        best_i = max(range(3), key=lambda i: agree_scores[i])
        best = cand[best_i]

        # optional fuse with summarizer
        if self.models.summarizer:
            try:
                fused = self.models.summarizer(a0 + "\n\n" + best, max_length=190, min_length=60)[0]["summary_text"].strip()
                if fused:
                    best = fused
            except Exception:
                pass

        meta = {
            "paa": True,
            "paa_prompts": [user_text, p1, p2],
            "paa_answer_pick": best_i,
            "paa_agreement": agree_scores,
            "base": m0,
            "p1": m1,
            "p2": m2,
        }
        return best, meta

    # c3: counterfactual consistency check (light)
    def _counterfactual_prompt(self, user_text: str) -> str:
        return user_text + " (précise aussi les hypothèses, limites, cas extrêmes, et une version 'si c'était faux')."

    def _c3_stage(self, user_text: str, answer: str) -> Tuple[str, Dict[str, Any]]:
        if not self.cfg.c3:
            return answer, {"c3": False}

        cfq = self._counterfactual_prompt(user_text)
        cf_ans, _ = self._consensus(cfq)

        agree = overlap_score(answer, cf_ans)
        meta = {"c3": True, "counterfactual_agreement": round(agree, 3)}

        # if disagreement, prepend assumptions disclaimer
        if agree < 0.30:
            prefix = (
                "note: il peut y avoir une ambiguïté dans la demande. "
                "je liste mes hypothèses, puis je réponds de façon robuste.\n"
                "- hypothèses: (1) contexte standard, (2) pas d'accès internet, (3) kb locale = source.\n\n"
            )
            # optional fuse to reduce contradiction
            if self.models.summarizer:
                try:
                    fused = self.models.summarizer(answer + "\n\n" + cf_ans, max_length=210, min_length=80)[0]["summary_text"].strip()
                    if fused:
                        return prefix + fused, meta
                except Exception:
                    pass
            return prefix + answer, meta

        return answer, meta

    # critic/governor: polish + reduce redundancy (offline)
    def _critic_stage(self, text: str) -> Tuple[str, Dict[str, Any]]:
        if not self.cfg.critic:
            return text, {"critic": False}

        rep = repetition_ratio(text)
        toks = tokenize_simple(text)
        length = len(toks)
        clarity = max(0.0, 1.0 - rep) * (1.0 if length > 12 else 0.7)
        meta = {"critic": True, "redundancy": round(rep, 3), "clarity": round(clarity, 3)}

        if clarity >= 0.78 and rep <= 0.20:
            return text, meta

        if self.models.summarizer:
            try:
                cleaned = self.models.summarizer(text, max_length=190, min_length=50)[0]["summary_text"].strip()
                if cleaned:
                    return cleaned, meta
            except Exception:
                pass

        # heuristic cleanup
        lines = sentence_split(text)
        dedup = []
        seen = set()
        for s in lines:
            key = normalize_ws(s).lower()
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return " ".join(dedup), meta

    # verification: per-sentence verdict backed by kb evidence + optional nli entailment
    def verify_answer(self, answer: str, k: int = 3) -> Tuple[str, List[ClaimVerdict], Dict[str, Any]]:
        sents = sentence_split(answer)
        if not sents or not self.kb.docs:
            return answer, [], {"verification": "skipped", "reason": "no sentences or empty kb"}

        kb_hash = self.kb.ledger_hash()
        verdicts: List[ClaimVerdict] = []
        supported = 0
        contradicted = 0

        for sent in sents:
            hits = self.kb.search(sent, k=k)
            evidence: List[EvidenceItem] = []
            verdict = "unverified"

            # if no hits: unverified
            if not hits:
                verdicts.append(ClaimVerdict(sentence=sent, verdict=verdict, evidence=[]))
                continue

            # attach evidence + optional nli
            best_ent = 0.0
            best_con = 0.0
            for h in hits:
                ent = None
                if self.cfg.verify_with_nli and self.models.nli:
                    ent = mnli_entailment_probs(self.models.nli, premise=h["text"], hypothesis=sent)
                    if ent:
                        best_ent = max(best_ent, ent.get("ENTAILMENT", 0.0))
                        best_con = max(best_con, ent.get("CONTRADICTION", 0.0))
                evidence.append(EvidenceItem(
                    doc_id=h["id"],
                    title=h["title"],
                    bm25_score=float(h["score"]),
                    entailment=ent
                ))

            # decide verdict
            if self.cfg.verify_with_nli and self.models.nli and (best_ent or best_con):
                if best_con >= self.cfg.nli_contradiction_threshold:
                    verdict = "contradicted"
                elif best_ent >= self.cfg.nli_supported_threshold:
                    verdict = "supported"
                else:
                    verdict = "unverified"
            else:
                # fallback: bm25 score threshold
                if evidence and max(e.bm25_score for e in evidence) >= self.cfg.bm25_min_score_for_claim:
                    verdict = "supported"
                else:
                    verdict = "unverified"

            if verdict == "supported":
                supported += 1
            if verdict == "contradicted":
                contradicted += 1

            verdicts.append(ClaimVerdict(sentence=sent, verdict=verdict, evidence=evidence))

        # annotate answer with tags
        tagged = []
        for v in verdicts:
            tag = f"(kb: {v.verdict})"
            tagged.append(v.sentence + " " + tag)
        tagged_text = " ".join(tagged)

        meta = {
            "verification": "done",
            "kb_hash": kb_hash,
            "supported_ratio": round(supported / max(1, len(verdicts)), 3),
            "contradicted_ratio": round(contradicted / max(1, len(verdicts)), 3),
            "nli_used": bool(self.cfg.verify_with_nli and self.models.nli),
        }
        return tagged_text, verdicts, meta

    def _maybe_translate(self, reply: str, user_lang: str) -> str:
        target = self.cfg.preferred_lang
        if target == "auto":
            target = user_lang or "fr"
        target = target.lower()

        if target.startswith("fr"):
            if self.models.trans_en_fr and self.detect_lang(reply) != "fr":
                try:
                    return self.models.trans_en_fr(reply)[0]["translation_text"]
                except Exception:
                    return reply
            return reply

        if target.startswith("en"):
            if self.models.trans_fr_en and self.detect_lang(reply) != "en":
                try:
                    return self.models.trans_fr_en(reply)[0]["translation_text"]
                except Exception:
                    return reply
            return reply

        return reply

    # -------------------------------------------------------------------------
    # commands
    # -------------------------------------------------------------------------

    def cmd_help(self) -> str:
        return (
            "commands:\n"
            "- /help\n"
            "- /about\n"
            "- /diag\n"
            "- /trace\n"
            "- /reset\n"
            "- /save [path]\n"
            "- /load [path]\n"
            "- /export [proofpack_path]\n"
            "- /set key=value (temperature, top_p, max_new_tokens, preferred_lang, consensus, n_consensus, paa, c3, critic, verify_with_nli)\n"
            "- /kb add <title>|<text>\n"
            "- /kb search <query>\n"
            "- /kb list\n"
            "- /kb clear\n"
            "- /contract set <json>\n"
            "- /contract show\n"
            "- /contract clear\n"
            "- /assert add <name>|<regex>\n"
            "- /assert list\n"
            "- /assert clear\n"
        )

    def cmd_about(self) -> str:
        info = {
            "device": str(self.device),
            "primary_model_loaded": bool(self.models.model_a),
            "secondary_model_loaded": bool(self.models.model_b),
            "translation_fr_en": bool(self.models.trans_fr_en),
            "translation_en_fr": bool(self.models.trans_en_fr),
            "summarizer": bool(self.models.summarizer),
            "nli_verifier": bool(self.models.nli),
            "kb_docs": len(self.kb.docs),
            "kb_hash": self.kb.ledger_hash() if self.kb.docs else None,
        }
        return "about:\n" + "\n".join(f"- {k}: {v}" for k, v in info.items())

    def cmd_diag(self) -> str:
        checks = [
            ("chat primary", bool(self.models.model_a)),
            ("chat secondary", bool(self.models.model_b)),
            ("paa translators", bool(self.models.trans_fr_en and self.models.trans_en_fr)),
            ("summarizer", bool(self.models.summarizer)),
            ("nli verifier", bool(self.models.nli)),
            ("kb docs", len(self.kb.docs)),
        ]
        return "diagnostic:\n" + "\n".join(f"- {k}: {('ok' if (v if isinstance(v,bool) else True) else 'missing') if isinstance(v,bool) else v}" for k, v in checks)

    def cmd_trace(self) -> str:
        if not self._last_trace:
            return "trace: none."
        return "trace:\n" + json.dumps(self._last_trace, ensure_ascii=False, indent=2)

    def cmd_reset(self) -> str:
        self.memory = ChatMemory(max_turns=self.memory.max_turns)
        self._last_trace = {}
        self._last_proofpack = None
        return "ok: memory reset."

    def cmd_save(self, path: Optional[str] = None) -> str:
        p = (path or "").strip() or self.cfg.state_path
        state = {
            "config": asdict(self.cfg),
            "turns": self.memory.to_json(),
            "kb_path": self.cfg.kb_path,
            "contract": asdict(self.contracts.contract) if self.contracts.contract else None,
            "assertions": self.contracts.assertions,
        }
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            # also persist kb
            self.kb.save(self.cfg.kb_path)
            return f"ok: saved state to {p} and kb to {self.cfg.kb_path}"
        except Exception as e:
            return f"error: save failed: {e}"

    def cmd_load(self, path: Optional[str] = None) -> str:
        p = (path or "").strip() or self.cfg.state_path
        try:
            with open(p, "r", encoding="utf-8") as f:
                state = json.load(f)
            cfg = EchoConfig(**(state.get("config") or {}))
            cfg.clamp()
            self.cfg = cfg
            self.memory.from_json(state.get("turns") or [])
            kb_path = state.get("kb_path") or self.cfg.kb_path
            if os.path.exists(kb_path):
                self.kb.load(kb_path)
            c = state.get("contract")
            if c:
                self.contracts.contract = ReplyContract(**c)
            self.contracts.assertions = state.get("assertions") or {}
            self._last_trace = {}
            return f"ok: loaded state from {p}"
        except Exception as e:
            return f"error: load failed: {e}"

    def cmd_export(self, path: Optional[str] = None) -> str:
        # export transcript + last proofpack if available
        proof_path = (path or "").strip() or self.cfg.proofpack_path
        try:
            # transcript
            with open(self.cfg.transcript_path, "w", encoding="utf-8") as f:
                f.write("# echoshield transcript\n\n")
                for t in self.memory.turns:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.ts))
                    f.write(f"**{t.speaker}** ({ts}):\n\n{t.text}\n\n---\n\n")

            # proof pack
            if self._last_proofpack:
                with open(proof_path, "w", encoding="utf-8") as f:
                    json.dump(self._last_proofpack, f, ensure_ascii=False, indent=2)

            return f"ok: exported transcript to {self.cfg.transcript_path} and proofpack to {proof_path}"
        except Exception as e:
            return f"error: export failed: {e}"

    def cmd_set(self, arg: str) -> str:
        if "=" not in arg:
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
            elif key == "preferred_lang":
                self.cfg.preferred_lang = val
            elif key == "consensus":
                self.cfg.consensus = val.lower() in {"1", "true", "yes", "y", "on"}
            elif key == "n_consensus":
                self.cfg.n_consensus = int(val)
            elif key == "paa":
                self.cfg.paa = val.lower() in {"1", "true", "yes", "y", "on"}
            elif key == "c3":
                self.cfg.c3 = val.lower() in {"1", "true", "yes", "y", "on"}
            elif key == "critic":
                self.cfg.critic = val.lower() in {"1", "true", "yes", "y", "on"}
            elif key == "verify_with_nli":
                self.cfg.verify_with_nli = val.lower() in {"1", "true", "yes", "y", "on"}
            else:
                return f"unknown key: {key}"
            self.cfg.clamp()
            return f"ok: {key} updated."
        except Exception as e:
            return f"error: invalid value for {key}: {e}"

    def cmd_kb(self, sub: str, rest: str) -> str:
        sub = (sub or "").strip().lower()
        rest = (rest or "").strip()

        if sub == "add":
            if "|" not in rest:
                return "format: /kb add title|text"
            title, text = rest.split("|", 1)
            title, text = title.strip(), text.strip()
            if not title or not text:
                return "format: /kb add title|text"
            doc_id = self.kb.add(title, text)
            # persist
            try:
                self.kb.save(self.cfg.kb_path)
            except Exception:
                pass
            return f"kb: added '{title}' (id={doc_id[:8]}..., total={len(self.kb.docs)})"

        if sub == "search":
            if not rest:
                return "format: /kb search query"
            hits = self.kb.search(rest, k=5)
            if not hits:
                return "kb: no results."
            lines = [f"- {h['title']} (score={h['score']:.3f})" for h in hits]
            return "kb results:\n" + "\n".join(lines)

        if sub == "list":
            if not self.kb.docs:
                return "kb: empty."
            return "kb docs:\n" + "\n".join(f"- {t}" for t in self.kb.list_titles())

        if sub == "clear":
            self.kb.clear()
            try:
                self.kb.save(self.cfg.kb_path)
            except Exception:
                pass
            return "kb: cleared."

        return "kb: unknown subcommand."

    def cmd_contract(self, sub: str, rest: str) -> str:
        sub = (sub or "").strip().lower()
        rest = (rest or "").strip()

        if sub == "set":
            try:
                obj = json.loads(rest)
                self.contracts.set_contract(obj)
                return "contract: set."
            except Exception as e:
                return f"contract: invalid json: {e}"

        if sub == "show":
            if not self.contracts.contract:
                return "contract: none."
            return "contract:\n" + json.dumps(asdict(self.contracts.contract), ensure_ascii=False, indent=2)

        if sub == "clear":
            self.contracts.clear_contract()
            return "contract: cleared."

        return "contract: unknown subcommand."

    def cmd_assert(self, sub: str, rest: str) -> str:
        sub = (sub or "").strip().lower()
        rest = (rest or "").strip()

        if sub == "add":
            if "|" not in rest:
                return "format: /assert add name|regex"
            name, regex = rest.split("|", 1)
            name, regex = name.strip(), regex.strip()
            if not name or not regex:
                return "format: /assert add name|regex"
            self.contracts.add_assertion(name, regex)
            return f"assert: added '{name}'."

        if sub == "list":
            if not self.contracts.assertions:
                return "assert: none."
            return "assert:\n" + "\n".join(f"- {k}: {v}" for k, v in self.contracts.assertions.items())

        if sub == "clear":
            self.contracts.clear_assertions()
            return "assert: cleared."

        return "assert: unknown subcommand."

    # -------------------------------------------------------------------------
    # main chat
    # -------------------------------------------------------------------------

    def _route(self, text: str) -> Optional[str]:
        if not text.startswith("/"):
            return None
        parts = text.split(" ", 2)
        cmd = parts[0].lower()

        if cmd == "/help":
            return self.cmd_help()
        if cmd == "/about":
            return self.cmd_about()
        if cmd == "/diag":
            return self.cmd_diag()
        if cmd == "/trace":
            return self.cmd_trace()
        if cmd == "/reset":
            return self.cmd_reset()
        if cmd == "/save":
            path = parts[1] if len(parts) > 1 else None
            return self.cmd_save(path)
        if cmd == "/load":
            path = parts[1] if len(parts) > 1 else None
            return self.cmd_load(path)
        if cmd == "/export":
            path = parts[1] if len(parts) > 1 else None
            return self.cmd_export(path)
        if cmd == "/set":
            if len(parts) < 2:
                return "format: /set key=value"
            return self.cmd_set(parts[1])
        if cmd == "/kb":
            if len(parts) < 2:
                return "kb: /kb add|search|list|clear ..."
            sub = parts[1]
            rest = parts[2] if len(parts) > 2 else ""
            return self.cmd_kb(sub, rest)
        if cmd == "/contract":
            if len(parts) < 2:
                return "contract: /contract set|show|clear ..."
            sub = parts[1]
            rest = parts[2] if len(parts) > 2 else ""
            return self.cmd_contract(sub, rest)
        if cmd == "/assert":
            if len(parts) < 2:
                return "assert: /assert add|list|clear ..."
            sub = parts[1]
            rest = parts[2] if len(parts) > 2 else ""
            return self.cmd_assert(sub, rest)

        return f"unknown command: {cmd}"

    def chat(self, user_input: str) -> str:
        if not self.limiter.allow():
            return "⏳ trop rapide — réessaie dans un instant."

        ui = normalize_ws(strip_control_and_invisibles(user_input or ""))
        if not ui:
            return "dis-moi quelque chose 🙂"

        # safety on input
        if not self.safety.is_safe(ui):
            return "🚫 contenu bloqué par le filtre de sécurité."

        # commands
        routed = self._route(ui)
        if routed is not None:
            return routed

        # store user
        self.memory.add("User", ui)

        # detect user language for final adaptation
        try:
            user_lang = langdetect.detect(ui)
        except Exception:
            user_lang = self.memory.last_user_lang() or "fr"

        # pipeline
        base, meta_paa = self._paa_stage(ui)
        refined, meta_c3 = self._c3_stage(ui, base)
        polished, meta_critic = self._critic_stage(refined)

        # contracts enforcement
        enforced = self.contracts.enforce(polished)

        # assertions check
        violations = self.contracts.check_assertions(enforced)

        # verification
        tagged, verdicts, meta_ver = self.verify_answer(enforced)

        # translate last
        final = self._maybe_translate(tagged, user_lang=user_lang)

        # update trace + proofpack
        self._last_trace = {
            "paa": meta_paa.get("paa"),
            "c3": meta_c3.get("c3"),
            "critic": meta_critic.get("critic"),
            "assert_violations": violations,
            "verification": meta_ver,
            "length_chars": len(final),
        }
        self._last_proofpack = {
            "ts": now_ts(),
            "user_input": ui,
            "final_answer": final,
            "trace": self._last_trace,
            "kb_hash": meta_ver.get("kb_hash"),
            "verdicts": [
                {
                    "sentence": v.sentence,
                    "verdict": v.verdict,
                    "evidence": [
                        {
                            "doc_id": e.doc_id,
                            "title": e.title,
                            "bm25_score": e.bm25_score,
                            "entailment": e.entailment,
                        }
                        for e in v.evidence
                    ],
                }
                for v in (verdicts or [])
            ],
        }

        # store bot
        self.memory.add("Bot", final)
        return final

# -----------------------------------------------------------------------------
# cli runner
# -----------------------------------------------------------------------------

def graceful_exit(signum, frame):
    print("\n👋 bye!")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, graceful_exit)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, graceful_exit)

    print(f"device: {get_device()}")
    try:
        bot = EchoShield()
    except Exception as e:
        logger.error(f"init failed: {e}")
        print("❌ impossible de démarrer: assure-toi que les modèles sont bien en cache local.")
        print("   astuce: lance un script de 'warmup' sur une machine connectée, puis copie ~/.cache/huggingface.")
        sys.exit(1)

    print("🤖 echoshield v2 prêt (offline). tape 'quit' pour sortir. /help pour l’aide.")
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
