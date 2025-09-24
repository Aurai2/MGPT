import os, re, sys, subprocess
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =============================
# Google Drive auto-download
# =============================
def install_and_import(package: str):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import("gdown")
import gdown  # noqa: E402

# Map artifact filenames to Google Drive file IDs  (these are yours; change if needed)
GDRIVE = {
    "phase2_best_rf.joblib": "1VYGciGkSXSZA8ispaOet0VlT359Tvwu4",
    "phase2_meta.joblib":    "12GxNNmk5qrWQZmL5lP36pjCEkNR-on_G",
    "phase1_features.parquet": "1xeS9XbxfDeEGBd8wIsgG7fzb5Ib8ky0c",
    "steel_dataset 1.csv":   "1yM2DAvkirtO0acUjWBPBRAjXl04guYp7",
    "rf_feature_importances.csv": "1_0dUQsfQdtjfIfZq6CTjc98VfulTEAh4",  # optional
}

# Download any missing files (kept as before)
for fname, fid in list(GDRIVE.items()):
    if fid and not Path(fname).exists():
        url = f"https://drive.google.com/uc?id={fid}"
        gdown.download(url, fname, quiet=False)

# =============================
# App config
# =============================
APP_TITLE = "SteelsGPT"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("SteelsGPT")
st.caption(
    "Use this interactive site to predict the properties of steels and ferrous alloys. "
    "This site utilises machine learning for forward predictions."
)

# Default file locations (same names you trained/exported)
MODEL_PATH   = os.environ.get("MODEL_PATH", "phase2_best_rf.joblib").strip()
META_PATH    = os.environ.get("META_PATH",  "phase2_meta.joblib").strip()
P1_PARQUET   = os.environ.get("P1_PARQUET", "phase1_features.parquet").strip()
REF_CSV      = os.environ.get("REF_CSV",    "steel_dataset 1.csv").strip()
FI_CSV       = os.environ.get("FI_CSV",     "rf_feature_importances.csv").strip()

# Composition schema (18 elements)
DEFAULT_COMP = {
    "Al": 0.00, "Cu": 0.00, "Mn": 1.00, "N": 0.00,
    "Ni": 0.30, "Ti": 0.00, "S": 0.00, "Fe": 97.00,
    "Zr": 0.00, "P": 0.00, "Si": 0.50, "V": 0.00,
    "Mo": 0.10, "Co": 0.00, "C": 0.50, "Nb": 0.00,
    "B": 0.00, "Cr": 0.50
}

# K=12 cluster legend (descriptions)
CLUSTERS_K12 = [
    {"code":0,  "label":"carburized – quenched (oil) – wear-resistant surface steel – temper"},
    {"code":1,  "label":"hot rolled – general construction steel – thickness"},
    {"code":2,  "label":"quenched (oil) – general-purpose steel – tested"},
    {"code":3,  "label":"normalized – easy-to-machine stock – cold drawn"},
    {"code":4,  "label":"normalized (air cooled) – general-purpose steel – round normalized"},
    {"code":5,  "label":"carburized – quenched (oil) – wear-resistant surface steel – quenched tempered"},
    {"code":6,  "label":"quenched (oil) & tempered – high-strength structural grade – oil"},
    {"code":7,  "label":"annealed – easy to machine – round annealed"},
    {"code":8,  "label":"carburized – quenched (oil) – wear-resistant surface steel – temper [Cluster 8]"},
    {"code":9,  "label":"normalized – general-purpose steel – round normalized"},
    {"code":10, "label":"quenched (water) & tempered – high-strength structural grade – water quenched"},
    {"code":11, "label":"Plate – no special treatment – general-purpose steel – plate grade"},
]
CODE2LABEL = {c["code"]: c["label"] for c in CLUSTERS_K12}

# -----------------------
# Helpers
# -----------------------
def load_model_and_meta(model_path: str, meta_path: str):
    if not Path(model_path).exists():
        st.error(f"Model file not found: {model_path}"); st.stop()
    if not Path(meta_path).exists():
        st.error(f"Meta file not found: {meta_path}"); st.stop()
    model = joblib.load(model_path)
    meta  = joblib.load(meta_path)
    features: List[str] = meta.get("features", [])
    targets:  List[str] = meta.get("targets", ["ultimate_tensile","yield","ductility"])
    if not features:
        st.error("Meta file missing `features`; cannot build the input vector."); st.stop()
    return model, features, targets

@st.cache_data(show_spinner=False)
def load_phase1_centroids(parquet_path: str, expected_p1_cols: List[str]):
    """Compute per-cluster centroids from the Phase-1 parquet (no file is modified)."""
    if not parquet_path or not Path(parquet_path).exists():
        return None, None, []
    p1 = pd.read_parquet(parquet_path)

    # Detect cluster id column
    clcol = None
    for cand in ["macro_cluster", "cluster_id", "cluster", "k_label"]:
        if cand in p1.columns:
            clcol = cand; break
    if clcol is None:
        return None, None, []

    # Filter to Phase-1 columns that the model expects
    p1_cols = [c for c in expected_p1_cols if c in p1.columns]

    # Ensure missing expected cols exist for mean calc
    for c in expected_p1_cols:
        if c not in p1.columns:
            p1[c] = 0.0

    cent = (p1.groupby(clcol)[expected_p1_cols]
              .mean()
              .reset_index()
              .rename(columns={clcol: "cluster_id"}))
    cluster_ids = cent["cluster_id"].tolist()
    return cent, cluster_ids, p1_cols

@st.cache_data(show_spinner=False)
def load_reference_csv(path: str):
    if not path or not Path(path).exists():
        return pd.DataFrame()
    # try common encodings
    df = None
    for enc in ["utf-8", "ISO-8859-1", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        return pd.DataFrame()
    # ensure name column
    if "alloy_name" not in df.columns:
        for alt in ["name","grade","label","steel_name","designation"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"alloy_name"}); break
    if "alloy_name" not in df.columns:
        df.insert(0, "alloy_name", [f"alloy_{i}" for i in range(len(df))])
    return df

def build_feature_row(feature_order: List[str],
                      comp: Dict[str, float],
                      cluster_id: int | None,
                      centroids: pd.DataFrame | None,
                      phase1_cols: List[str]):
    """Build the exact model input: composition + Phase-1 features and cl_* one-hots."""
    row = {f: float(comp.get(f, 0.0)) for f in feature_order}

    # cl_* one-hots if present in model features
    cl_cols = [c for c in feature_order if c.startswith("cl_")]
    if cl_cols:
        for c in cl_cols:
            row[c] = 0.0
        if cluster_id is not None and f"cl_{int(cluster_id)}" in cl_cols:
            row[f"cl_{int(cluster_id)}"] = 1.0

    # Phase-1 centroid injection for route__/tag__/form__/proc_pca_* expected by model
    if centroids is not None and phase1_cols:
        if cluster_id is None and "cluster_id" in centroids.columns and len(centroids):
            sel = centroids.iloc[[0]]
        else:
            sel = centroids.loc[centroids["cluster_id"] == int(cluster_id)] if "cluster_id" in centroids.columns else pd.DataFrame()
            if sel.empty and len(centroids):
                sel = centroids.iloc[[0]]
        for c in phase1_cols:
            if c in sel.columns:
                row[c] = float(sel.iloc[0][c])

    return row

# ---- process text search (for suggested alloy ranking) ----
def tokenize_query(q: str) -> List[str]:
    parts = re.findall(r'"([^"]+)"|(\S+)', q)
    terms = [p[0] or p[1] for p in parts]
    return [t.strip() for t in terms if t.strip()]

def build_regex(terms: List[str], mode: str) -> re.Pattern | None:
    if not terms: return None
    esc = [re.escape(t) for t in terms]
    if mode == "OR":
        return re.compile(r"(?i)(" + "|".join(esc) + r")")
    # AND requires all via lookaheads
    return re.compile(r"(?i)" + "".join([f"(?=.*{e})" for e in esc]))

def excerpt(text: str, pattern: re.Pattern, radius: int = 50) -> str:
    m = pattern.search(text)
    if not m:
        return (text[:2*radius] + "…") if len(text) > 2*radius else text
    start = max(0, m.start() - radius)
    end   = min(len(text), m.end() + radius)
    out = text[start:end]
    return ("…" if start > 0 else "") + out + ("…" if end < len(text) else "")

def rank_suggestions(ref_df: pd.DataFrame,
                     comp_input: Dict[str, float],
                     pred_props: Dict[str, float] | None,
                     process_text_col: str | None,
                     query: str,
                     query_mode: str,
                     topk: int = 10):
    """Rank alloys by: text match + composition closeness + (optional) property closeness."""
    if ref_df.empty:
        return pd.DataFrame()

    df = ref_df.copy()

    # --- text score ---
    text_score = np.zeros(len(df), dtype=float)
    if process_text_col and process_text_col in df.columns and query.strip():
        terms = tokenize_query(query)
        rx = build_regex(terms, query_mode)
        tx = df[process_text_col].fillna("").astype(str)
        if rx:
            nums = re.findall(r"\d+\.?\d*", query)  # boost exact numeric mentions
            def score_txt(txt):
                hits = [m.group(0) for m in re.finditer(re.compile("|".join([re.escape(t) for t in terms]), re.I), txt)]
                base = sum(len(h) for h in hits)
                boost = sum(100 for n in nums if re.search(rf"(?i)\b{re.escape(n)}\b", txt))
                return base + boost
            text_score = tx.apply(score_txt).to_numpy()
            df["match_excerpt"] = tx.apply(lambda t: excerpt(t, rx))
    else:
        df["match_excerpt"] = ""

    # --- composition distance ---
    comp_cols = [c for c in DEFAULT_COMP if c in df.columns]
    comp_dist = np.zeros(len(df), dtype=float)
    if comp_cols:
        v = np.array([float(comp_input.get(c, 0.0)) for c in comp_cols], dtype=float)
        M = df[comp_cols].to_numpy(dtype=float)
        comp_dist = np.linalg.norm(M - v[None, :], axis=1)
    comp_score = 1.0 - ((comp_dist - comp_dist.min()) / (comp_dist.max() - comp_dist.min() + 1e-12))

    # --- property distance (optional), uses predicted props
    prop_cols = [c for c in ["ultimate_tensile","yield","ductility"] if c in df.columns]
    prop_score = np.zeros(len(df), dtype=float)
    if pred_props and prop_cols and all(k in pred_props for k in ["ultimate_tensile","yield","ductility"]):
        v = np.array([float(pred_props.get(k, 0.0)) for k in prop_cols], dtype=float)
        M = df[prop_cols].to_numpy(dtype=float)
        d = np.linalg.norm(M - v[None, :], axis=1)
        prop_score = 1.0 - ((d - d.min()) / (d.max() - d.min() + 1e-12))

    # Final score
    if query.strip():
        score = 0.60 * (text_score / (text_score.max() or 1.0)) + 0.25 * comp_score + 0.15 * prop_score
    else:
        score = 0.60 * comp_score + 0.40 * prop_score

    out = df.copy()
    out["_score"] = score
    out = out.sort_values("_score", ascending=False).head(topk)

    # Choose preview columns
    show_cols = []
    process_display_col = None
    for c in ["process","route","notes","treatment","processing","description"]:
        if c in out.columns: process_display_col = c; break
    for c in ["alloy_name", process_display_col, "match_excerpt", "_score",
              "ultimate_tensile","yield","ductility"]:
        if c and c in out.columns and c not in show_cols:
            show_cols.append(c)
    comp_preview = [c for c in ["C","Si","Mn","Cr","Ni","Mo","V","Nb","Ti","Al","Cu","N","B","W","Co","Fe","Zr","P","S"]
                    if c in out.columns][:8]
    show_cols += [c for c in comp_preview if c not in show_cols]
    return out[show_cols]

# -----------------------
# Load artifacts
# -----------------------
model, feature_order, target_names = load_model_and_meta(MODEL_PATH, META_PATH)

# Determine which Phase-1 features the model expects
PHASE1_PREFIXES = ("cl_","route__","tag_","form_","proc_pca_")
p1_expected = [c for c in feature_order if c.startswith(PHASE1_PREFIXES)]
centroids, cluster_ids, phase1_cols = load_phase1_centroids(P1_PARQUET, p1_expected)

# Reference dataset (for suggested alloys + process text ranking)
ref_df = load_reference_csv(REF_CSV)

# Optional feature importances
fi_df = pd.read_csv(FI_CSV) if Path(FI_CSV).exists() else pd.DataFrame()

# -----------------------
# UI
# -----------------------
with st.sidebar:
    st.subheader("Files")
    st.caption(f"Model: {MODEL_PATH}")
    st.caption(f"Meta: {META_PATH}")
    st.caption(f"Phase-1 parquet: {P1_PARQUET}")
    st.caption(f"Reference CSV: {REF_CSV}")
    if not p1_expected:
        st.info("Model appears to be composition-only (no Phase-1 feature names found). "
                "Predictions will not depend on processing cluster.")
    else:
        st.caption(f"Phase-1 feature count expected by model: {len(p1_expected)}")

left, right = st.columns([0.62, 0.38])

with left:
    # ---- Composition (FIRST) ----
    st.subheader("Select alloy composition, in wt. %")
    st.caption("(*note, the Fe content will automatically balance the composition to 100%*)")

    cols = st.columns(6)
    comp = {}
    for i, (elem, default) in enumerate(DEFAULT_COMP.items()):
        with cols[i % 6]:
            comp[elem] = st.number_input(elem, min_value=0.0, max_value=100.0,
                                         value=float(default), step=0.01)

    # Fe auto-balance preview
    sum_except_fe = sum(v for k, v in comp.items() if k != "Fe")
    fe_bal = max(0.0, 100.0 - sum_except_fe)
    st.caption(f"Fe will be set to **{fe_bal:.2f} wt.%** to balance to 100%.")

    # ---- Processing (SECOND) ----
    st.subheader("Select alloy processing condition")
    use_cluster = st.checkbox("Uncheck if you do not wish to use alloy processing as an input", value=True)
    cluster_id = None
    if use_cluster:
        cluster_id = st.selectbox(
            "Cluster (K=12)",
            [c["code"] for c in CLUSTERS_K12],
            index=1,
            format_func=lambda cid: f"{cid} — {CODE2LABEL.get(cid, 'Cluster')}"
        )
        st.caption(f"Selected: **Cluster {cluster_id} — {CODE2LABEL.get(cluster_id, '—')}**")

    # ---- (OPTIONAL) custom process text ----
    st.subheader("(optional) Process text (user entries possible for processing)")
    st.caption("Use drop down processing selection, or user entry for processing")
    process_query = st.text_input('', "", placeholder='e.g. "cold drawn" 870 degrees pseudo-carburized')
    # keep the radio (AND/OR) but hide the label
    query_mode = st.radio("", ["AND","OR"], index=0, horizontal=True)

    if st.button("Predict", type="primary"):
        # enforce Fe balance on submit
        comp["Fe"] = fe_bal

        # Build model input row in the exact saved feature order
        row = build_feature_row(
            feature_order, comp,
            cluster_id if use_cluster else None,
            centroids if use_cluster else None,
            phase1_cols
        )
        X = pd.DataFrame([row], columns=feature_order).astype(float)

        # Predict
        y_pred = model.predict(X)
        y_vec = np.array(y_pred).reshape(-1)
        preds = {t: float(v) for t, v in zip(target_names, y_vec)}

        # Persist for right panel + suggestions
        st.session_state["preds"] = preds
        st.session_state["comp"] = comp
        st.session_state["process_query"] = process_query
        st.session_state["query_mode"] = query_mode

with right:
    st.subheader("Results")
    preds = st.session_state.get("preds")
    if preds is None:
        st.info("Set inputs and click **Predict**.")
    else:
        # Show targets (1 decimal place)
        cols = st.columns(len(target_names))
        for i, t in enumerate(target_names):
            with cols[i]:
                val = preds.get(t, None)
                st.metric(t, f"{val:.1f}" if isinstance(val, (int, float)) else f"{val}")

        # Suggested alloys (ranked)
        st.markdown("### Suggested matching alloys from training database")
        if ref_df.empty:
            st.info("Reference CSV not found or empty. Place your dataset at `steel_dataset 1.csv` or set REF_CSV.")
        else:
            # Pick the best process text column (if any)
            proc_col = None
            for c in ["process","route","notes","treatment","processing","description"]:
                if c in ref_df.columns:
                    proc_col = c; break

            ranked = rank_suggestions(
                ref_df,
                st.session_state.get("comp", {}),
                preds,
                proc_col,
                st.session_state.get("process_query", ""),
                st.session_state.get("query_mode", "AND"),
                topk=10
            )
            if ranked.empty:
                st.info("No suggestions match your filters. Try adjusting the process text or ensure expected columns exist in the CSV.")
            else:
                st.dataframe(ranked, use_container_width=True)
                st.download_button(
                    "⬇️ Download suggestions",
                    ranked.to_csv(index=False).encode("utf-8"),
                    "suggested_alloys.csv",
                    "text/csv"
                )

# Optional: feature importances expander
if Path(FI_CSV).exists():
    try:
        fi_df = pd.read_csv(FI_CSV)
        with st.expander("Feature importances (from Phase-2 RF)"):
            st.dataframe(fi_df.head(50), use_container_width=True)
    except Exception:
        pass

