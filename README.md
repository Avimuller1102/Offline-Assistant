# Offline-Assistant
A fully offline, multilingual, self-consensus AI chatbot with built-in knowledge verification, reflexive critic, and safety guard. No cloud. No dependencies. 100% local intelligence.

# EchoShield — offline self-consensus chatbot (paa + c3 + rcg, contracts, kb)

אקו־שילד הוא צ’אטבוט מקומי, אולטרה-אמין ו-100% אופליין: הוא מייצר כמה טיוטות תשובה, משיג **קונצנזוס עצמי** ביניהן (tscd), בודק עקביות בעזרת **פרפרזה** (paa) ו-**שאלה נגדית** (c3), מעביר ביקורת עצמית (rcg), ומסמן כל משפט כ-**נתמך / לא מאומת** מול **מאגר ידע מקומי**. בנוסף, הוא תומך ב-**חוזי תשובה** ו-**אישורי משתמש** (regex) לאכיפה קשיחה, ובולם prompt-injection — והכל ללא רשת.

EchoShield is a tiny, pure-Python, **fully offline** chatbot that runs locally with no network calls. It:
- generates multiple candidate answers and performs **triple self-consensus decoding (tscd)**,
- does **paraphrase-and-agree (paa)** and a **counterfactual consistency check (c3)**,
- passes a **reflexive critic & governor (rcg)** to polish clarity and reduce redundancy,
- tags each sentence as **(kb: supported)** or **(kb: unverified)** using a local TF-IDF mini-KB,
- enforces **reply contracts** and **user assertions** (regex) on the final text,
- ships with a strict **safety filter** and **prompt-injection shield**,
- works with your **local HuggingFace cache only** (no internet required).

**no internet. no daemons. single python file. comments in english and in lowercase only.**

---

## Quick Start

> **requirements (offline)**:  
> python 3.9+ · torch · transformers · langdetect  · locally cached hf models:  
> `facebook/blenderbot-400M-distill`, `facebook/mbart-large-50-many-to-many-mmt`,  
> `Helsinki-NLP/opus-mt-fr-en`, `Helsinki-NLP/opus-mt-en-fr`,  
> `facebook/bart-large-cnn`, `distilbert-base-cased-distilled-squad`.

```bash
# run the chatbot locally (no internet)
python offline_assistant.py
```

example session:

```
➡️ vous : /diag
💬 bot : diagnostic:
- primary model: ok
- translator fr→en: ok
…

➡️ vous : /kb add guide|le guide interne décrit l'api x et ses limites.
💬 bot : kb: ajouté « guide » (total: 1)

➡️ vous : comment utiliser l'api x en mode offline ?
💬 bot : …
… (kb: supported) … (kb: unverified)
```

---

## Files & deps

- **single file**: `offline_assistant.py` (the main script)  
- **optional**: `cryptography` (only if you later add signing to exports)  
- **runtime libs**: `torch`, `transformers`, `langdetect`  
- **models (cached locally)**:  
  - `facebook/blenderbot-400M-distill`  
  - `facebook/mbart-large-50-many-to-many-mmt`  
  - `Helsinki-NLP/opus-mt-fr-en`, `Helsinki-NLP/opus-mt-en-fr`  
  - `facebook/bart-large-cnn`  
  - `distilbert-base-cased-distilled-squad`

---

**ready to go** — run `python offline_assistant.py`, add a few `/kb` snippets, set a `/contract`, and chat.  
you’ll get **clean, robust, tagged** answers — entirely **offline**.
