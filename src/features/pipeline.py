"""
Assembles all feature extractors into a single sklearn-compatible pipeline.

Feature groups
--------------
1. Hand-crafted text scalars  (11 features)  — TextFeatureTransformer on 'body'
2. URL signals                ( 6 features)  — UrlFeatureTransformer on ['body','html_body']
3. Structural HTML signals    ( 8 features)  — StructuralFeatureTransformer on ['html_body','body']
4. Header anomaly signals     ( 5 features)  — HeaderFeatureTransformer on ['sender','reply_to']
5. TF-IDF + TruncatedSVD      (100 features) — on concatenated subject + body

Total: 130 dense float features per email.

Usage
-----
    from src.features.pipeline import build_feature_pipeline
    import pandas as pd

    df = pd.read_csv("data/processed/cleaned.csv")
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df)   # numpy array, shape (n, 130)
    y = df["label"].values
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.header_extractor import HeaderFeatureTransformer
from src.features.structural_extractor import StructuralFeatureTransformer
from src.features.text_extractor import TextFeatureTransformer
from src.features.url_extractor import UrlFeatureTransformer

N_SVD_COMPONENTS = 100


# ---------------------------------------------------------------------------
# Helper: combine subject + body for TF-IDF
# ---------------------------------------------------------------------------

class SubjectBodyCombiner(BaseEstimator, TransformerMixin):
    """
    Combines the 'subject' and 'body' columns into a single string per row
    for TF-IDF vectorisation.

    Expects a DataFrame with columns 'subject' and 'body'.
    Returns a list of strings (subject \\n body).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> list[str]:
        if hasattr(X, "iloc"):
            subjects = X["subject"].fillna("").astype(str)
            bodies = X["body"].fillna("").astype(str)
        else:
            subjects = [str(r[0]) for r in X]
            bodies = [str(r[1]) for r in X]
            return [f"{s}\n{b}" for s, b in zip(subjects, bodies)]
        return [f"{s}\n{b}" for s, b in zip(subjects, bodies)]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_feature_pipeline() -> Pipeline:
    """
    Build and return the full feature extraction pipeline.

    The returned Pipeline accepts a pandas DataFrame with at minimum the
    columns: subject, body, html_body, sender, reply_to.

    Call .fit_transform(df) on the training set, then .transform(df) on
    held-out sets. Output is a dense float64 numpy array.
    """

    tfidf_svd = Pipeline([
        ("combiner", SubjectBodyCombiner()),
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            max_features=15_000,
            ngram_range=(1, 2),
            min_df=3,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]{1,}\b',
        )),
        ("svd", TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=42)),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            (
                "text_feats",
                TextFeatureTransformer(),
                "body",                          # Series[str]
            ),
            (
                "url_feats",
                UrlFeatureTransformer(),
                ["body", "html_body"],            # DataFrame
            ),
            (
                "struct_feats",
                StructuralFeatureTransformer(),
                ["html_body", "body"],            # DataFrame
            ),
            (
                "header_feats",
                HeaderFeatureTransformer(),
                ["sender", "reply_to"],           # DataFrame
            ),
            (
                "tfidf_svd",
                tfidf_svd,
                ["subject", "body"],              # DataFrame → SubjectBodyCombiner → str list
            ),
        ],
        remainder="drop",
        n_jobs=1,
    )

    # Wrap in a Pipeline. StandardScaler normalises the hand-crafted scalars
    # alongside the SVD components so that tree models and linear models both
    # receive well-conditioned input without extra effort.
    return Pipeline([
        ("features", col_transformer),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Feature name utility
# ---------------------------------------------------------------------------

def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Return a list of human-readable feature names for a fitted pipeline.
    Useful for feature importance plots.
    """
    col_transformer: ColumnTransformer = pipeline.named_steps["features"]
    names: list[str] = []
    for name, transformer, _ in col_transformer.transformers_:
        if name == "remainder":
            continue
        if isinstance(transformer, Pipeline):
            # TF-IDF + SVD sub-pipeline: generate generic SVD component names
            svd: TruncatedSVD = transformer.named_steps["svd"]
            names.extend([f"svd_{i}" for i in range(svd.n_components)])
        elif hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out().tolist())
        else:
            names.append(name)
    return names
