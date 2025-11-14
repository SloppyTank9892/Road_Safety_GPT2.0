#!/usr/bin/env python3
"""
rs_gpt.py
Enhanced Road Safety recommender (single-name edition).
Features:
 - TF-IDF matching (always)
 - Optional sentence-transformer embeddings (if installed)
 - Rule-based context boosts (night, rain, pedestrian, highway...)
 - Lightweight local 'why' explanations
 - Feedback (thumbs-up) persistence
 - Saves/loads via CSV + joblib (no parquet)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
import re, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: sentence-transformers (install separately if you want embeddings)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_ST = True
except Exception:
    _HAS_ST = False

REQUIRED_COLUMNS = ["problem", "category", "type", "data", "code", "clause"]

# Rule lexicons (expandable)
RULES = {
    "environment": {
        "night": ["night", "dark", "low light", "visibility"],
        "rain": ["rain", "wet", "rainy"],
        "fog": ["fog", "mist"],
        "work_zone": ["work zone", "construction"],
    },
    "road_type": {
        "highway": ["highway", "expressway", "nh", "freeway"],
        "urban": ["urban", "city", "arterial"],
        "rural": ["rural", "village", "two-lane", "two lane"],
        "intersection": ["intersection", "junction", "roundabout"],
        "school_zone": ["school", "school zone"],
    },
    "user": {
        "pedestrian": ["pedestrian", "zebra", "crosswalk", "foot"],
        "two_wheeler": ["two-wheeler", "motorcycle", "scooter", "bike"],
        "bicycle": ["bicycle", "cyclist", "cycle"],
        "bus": ["bus"],
        "truck": ["truck", "lorry"],
        "car": ["car", "four-wheeler"],
    }
}

@dataclass
class RSConfig:
    # TF-IDF parameters
    ngram_min: int = 1
    ngram_max: int = 4  # Increased to catch longer phrases
    stop_words: str = "english"
    min_df: int = 1
    max_df: float = 0.90  # Ignore terms that appear in >90% of docs

    # Weights for different fields (reserved for future use)
    problem_weight: float = 2.0
    category_weight: float = 2.5
    type_weight: float = 3.5
    data_weight: float = 1.0

    # Embedding model
    embed_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    enable_embeddings: bool = True

    # rule boost
    rule_boost_per_hit: float = 0.08

    # Hybrid scoring weights (can be tuned live)
    w_tfidf: float = 0.7
    w_embed: float = 0.2
    w_rules: float = 0.1
    w_feedback: float = 0.0
    # Exact-match token boost weight
    w_exact: float = 0.06
    

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _extract_hits(text: str):
    low = _normalize_text(text)
    hits = {k: [] for k in RULES}
    for group, tags in RULES.items():
        for tag, words in tags.items():
            if any(w in low for w in words):
                hits[group].append(tag)
    return hits

class RSInterventionGPT:
    def __init__(self, cfg: RSConfig | None = None):
        self.cfg = cfg or RSConfig()
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_mat = None
        self.st_model = None
        self.doc_emb = None
        self.feedback_counts: Dict[str, int] = {}

    def _load_df(self, db_path: str | Path) -> pd.DataFrame:
        p = Path(db_path)
        if not p.exists():
            raise FileNotFoundError(f"Database not found: {p}")
        if p.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")
        return df.copy()

    def _fulltext_series(self, df: pd.DataFrame):
        return (df["problem"].astype(str) + " " + df["category"].astype(str) + " " +
                df["type"].astype(str) + " " + df["data"].astype(str))

    def fit(self, db_path: str | Path):
        """Train TF-IDF and (optionally) document embeddings."""
        self.df = self._load_df(db_path)
        self.df["_fulltext"] = self._fulltext_series(self.df)

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words=self.cfg.stop_words,
            ngram_range=(self.cfg.ngram_min, self.cfg.ngram_max),
            min_df=self.cfg.min_df,
            max_df=getattr(self.cfg, 'max_df', 0.95),
            sublinear_tf=True,
            norm='l2'
        )
        self.doc_mat = self.vectorizer.fit_transform(self.df["_fulltext"])

        # Embeddings (optional)
        if self.cfg.enable_embeddings and _HAS_ST:
            self.st_model = SentenceTransformer(self.cfg.embed_model_name)
            self.doc_emb = self.st_model.encode(self.df["_fulltext"].tolist(), convert_to_tensor=True, show_progress_bar=False)

        return self

    def save(self, model_dir: str | Path):
        """Save vectorizer, doc matrix, table, optional embeddings, feedback."""
        if self.df is None or self.vectorizer is None or self.doc_mat is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, model_dir / "vectorizer.joblib")
        joblib.dump(self.doc_mat, model_dir / "doc_mat.joblib")
        self.df.to_csv(model_dir / "interventions.csv", index=False)
        if self.doc_emb is not None:
            joblib.dump(self.doc_emb, model_dir / "doc_emb.joblib")
            joblib.dump({"model_name": self.cfg.embed_model_name}, model_dir / "embed_meta.joblib")
        joblib.dump(self.feedback_counts, model_dir / "feedback_counts.joblib")

    @classmethod
    def load(cls, model_dir: str | Path) -> "RSInterventionGPT":
        model_dir = Path(model_dir)
        obj = cls()
        obj.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        obj.doc_mat = joblib.load(model_dir / "doc_mat.joblib")
        obj.df = pd.read_csv(model_dir / "interventions.csv")
        # Try load embeddings
        try:
            obj.doc_emb = joblib.load(model_dir / "doc_emb.joblib")
            meta = joblib.load(model_dir / "embed_meta.joblib")
            if _HAS_ST:
                obj.st_model = SentenceTransformer(meta.get("model_name", obj.cfg.embed_model_name))
        except Exception:
            obj.doc_emb = None
            obj.st_model = None
        # Feedback
        try:
            obj.feedback_counts = joblib.load(model_dir / "feedback_counts.joblib")
        except Exception:
            obj.feedback_counts = {}
        return obj

    # scoring helpers
    def _score_tfidf(self, query: str, idxs):
        qv = self.vectorizer.transform([query])
        sim = cosine_similarity(qv, self.doc_mat[idxs]).ravel()
        return sim

    def _score_field(self, query: str, view: pd.DataFrame, field: str):
        """Compute cosine similarity between query and a specific text field for each row in view."""
        texts = view[field].astype(str).tolist()
        if len(texts) == 0:
            return np.array([])
        qv = self.vectorizer.transform([query])
        fv = self.vectorizer.transform(texts)
        sim = cosine_similarity(qv, fv).ravel()
        # Normalize field sim if non-constant
        if sim.max() - sim.min() >= 1e-9:
            sim = (sim - sim.min()) / (sim.max() - sim.min())
        return sim

    def _score_embed(self, query: str, idxs):
        if self.st_model is None or self.doc_emb is None:
            return np.zeros(len(idxs), dtype=float)
        q_emb = self.st_model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        sim = st_util.cos_sim(q_emb, self.doc_emb[idxs]).cpu().numpy().ravel()
        if sim.max() - sim.min() < 1e-9:
            return np.zeros_like(sim)
        return (sim - sim.min()) / (sim.max() - sim.min())

    def _score_rules(self, query: str, view: pd.DataFrame):
        hits = _extract_hits(query)
        boost = np.zeros(len(view), dtype=float)
        if not any(hits.values()):
            return boost
        for i, (_, row) in enumerate(view.iterrows()):
            row_text = (" ".join([str(row.get("problem","")), str(row.get("category","")),
                                  str(row.get("type","")), str(row.get("data",""))])).lower()
            row_boost = 0.0
            for group, tags in hits.items():
                for tag in tags:
                    if tag in row_text:
                        row_boost += self.cfg.rule_boost_per_hit
            boost[i] = row_boost
        return boost

    def _score_feedback(self, view: pd.DataFrame):
        keys = (view["problem"].astype(str) + " || " + view["type"].astype(str)).tolist()
        counts = np.array([self.feedback_counts.get(k, 0) for k in keys], dtype=float)
        if counts.max() == 0:
            return np.zeros_like(counts)
        return counts / counts.max()

    def _score_exact(self, query: str, view: pd.DataFrame):
        """Simple exact-token match boost across problem/category/type fields."""
        q_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        if not q_tokens:
            return np.zeros(len(view), dtype=float)
        boosts = np.zeros(len(view), dtype=float)
        # count matches per field and weight them by field importance
        for i, (_, row) in enumerate(view.iterrows()):
            cnt = 0.0
            for field, w in [('problem', self.cfg.problem_weight), ('category', self.cfg.category_weight), ('type', self.cfg.type_weight)]:
                text = str(row.get(field, '')).lower()
                cnt_field = sum(1 for t in q_tokens if f" {t} " in f" {text} ")
                cnt += cnt_field * float(w)
            boosts[i] = cnt
        if boosts.max() <= 0:
            return np.zeros_like(boosts)
        return boosts / boosts.max()

    def _hybrid_score(self, query: str, view_idx, view_df):
        # Use per-field TF-IDF similarities and combine with configured weights
        s_problem = self._score_field(query, view_df, 'problem')
        s_category = self._score_field(query, view_df, 'category')
        s_type = self._score_field(query, view_df, 'type')
        s_data = self._score_field(query, view_df, 'data')

        # Handle potential empty fields
        weights = np.array([
            self.cfg.problem_weight,
            self.cfg.category_weight,
            self.cfg.type_weight,
            self.cfg.data_weight
        ], dtype=float)

        # Stack and compute weighted average
        stacked = np.vstack([
            s_problem,
            s_category,
            s_type,
            s_data
        ])
        # If some fields returned empty arrays, replace with zeros
        if stacked.shape[1] != len(view_df):
            # fallback to overall TF-IDF similarity
            s_tfidf = self._score_tfidf(query, view_idx)
            if s_tfidf.max() - s_tfidf.min() >= 1e-9:
                s_tfidf = (s_tfidf - s_tfidf.min()) / (s_tfidf.max() - s_tfidf.min())
        else:
            weighted = (weights[:, None] * stacked).sum(axis=0) / (weights.sum())
            s_tfidf = weighted
        s_embed = self._score_embed(query, view_idx)
        s_rules = self._score_rules(query, view_df)
        s_fb = self._score_feedback(view_df)
        s_exact = self._score_exact(query, view_df)

        # Weighted combination
        score = (self.cfg.w_tfidf * s_tfidf +
                 self.cfg.w_embed * s_embed +
                 self.cfg.w_rules * s_rules +
                 self.cfg.w_feedback * s_fb +
                 self.cfg.w_exact * s_exact)

        # Normalize to 0..1
        if score.max() - score.min() >= 1e-9:
            score = (score - score.min()) / (score.max() - score.min())

        # Stretch top scores slightly to increase contrast (milder)
        stretch = 1.2
        score = np.power(score, stretch)

        return score

    def recommend(self, query: str, top_k: int = 5, category: Optional[str] = None, type_filter: Optional[str] = None):
        if self.df is None or self.vectorizer is None or self.doc_mat is None:
            raise RuntimeError("Model not loaded/trained.")
        view = self.df
        if category:
            view = view[view["category"].astype(str).str.contains(category, case=False, na=False)]
        if type_filter:
            view = view[view["type"].astype(str).str.contains(type_filter, case=False, na=False)]
        if view.empty:
            return pd.DataFrame(columns=["rank","score","problem","category","type","explanation","reference_code","reference_clause","why"])
        idxs = view.index.to_numpy()
        hybrid = self._hybrid_score(query, idxs, view)
        order = np.argsort(-hybrid)[:top_k]
        rows = []
        for i, pos in enumerate(order, start=1):
            row = view.iloc[pos]
            rows.append({
                "rank": i,
                "score": round(float(hybrid[pos]), 4),
                "problem": row["problem"],
                "category": row["category"],
                "type": row["type"],
                "explanation": row["data"],
                "reference_code": row["code"],
                "reference_clause": row["clause"],
                "why": self._reason(query, row)
            })
        return pd.DataFrame(rows)

    def record_feedback(self, item_row: pd.Series, positive: bool = True):
        if not positive:
            return
        key = f"{item_row.get('problem','')} || {item_row.get('type','')}"
        self.feedback_counts[key] = self.feedback_counts.get(key, 0) + 1

    def _reason(self, query: str, row: pd.Series) -> str:
        hits = _extract_hits(query)
        parts = []
        hit_bits = []
        for grp, tags in hits.items():
            if tags:
                pretty = ", ".join(tags).replace("_", " ")
                hit_bits.append(f"{grp}: {pretty}")
        if hit_bits:
            parts.append(f"Context cues detected → { '; '.join(hit_bits) }.")
        tie = []
        for label, key in [("Problem","problem"), ("Category","category"), ("Type","type")]:
            val = str(row.get(key,"")).strip()
            if val:
                tie.append(f"{label}: {val}")
        if tie:
            parts.append("Matches entry → " + " | ".join(tie))
        code = str(row.get("code","")).strip()
        clause = str(row.get("clause","")).strip()
        if code or clause:
            parts.append(f"Reference: {code} Clause {clause}")
        data = str(row.get("data","")).strip()
        if data:
            parts.append(f"Details: {data}")
        return " ".join(parts) if parts else "Recommended based on similarity to your issue."
