
import streamlit as st
import pandas as pd
from pathlib import Path

from utils import load_csv_safely
from backend import prepare_backend
from recommend import recommend_alloys
from search import semantic_search

st.set_page_config(page_title="MaterialsGPT – Alloy Finder", layout="wide")
st.title("MaterialsGPT – Alloy Finder")

st.markdown("""
Upload your dataset once. The app will **prepare in the background** (Phase‑1 + train models).
Users only need to **enter desired properties** to get recommended alloys.
""")

with st.sidebar:
    st.header("Setup")
    process_col = st.text_input("Processing text column", value="process")
    id_col = st.text_input("ID column (optional)", value="row_id")
    composition_prefix = st.text_input("Composition prefix", value="pct_")
    embed_model = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"], index=0)
    birch_threshold = st.slider("BIRCH threshold", 0.05, 1.0, 0.25, 0.05)
    agglo_clusters = st.number_input("Agglomerative clusters", 2, 64, 12, 1)

uploaded = st.file_uploader("Upload your steel dataset CSV", type=["csv"])

@st.cache_resource(show_spinner=False)
def _prepare(csv_path, process_col, id_col, composition_prefix, embed_model, birch_threshold, agglo_clusters):
    return prepare_backend(
        csv_path=csv_path,
        process_col=process_col,
        id_col=id_col if id_col else None,
        features_out="phase1_features.parquet",
        embed_model=embed_model,
        birch_threshold=float(birch_threshold),
        agglo_clusters=int(agglo_clusters),
        composition_prefix=composition_prefix or None,
        target_cols=None,
    )

if uploaded is None:
    st.info("Upload a CSV to initialize the backend.")
else:
    csv_path = Path("uploaded.csv").resolve()
    csv_path.write_bytes(uploaded.getbuffer())
    df = load_csv_safely(csv_path)
    st.success(f"CSV uploaded: {csv_path.name}")
    st.write("Preview:", df.head(10))

    # Auto‑prepare backend (Phase‑1 + train models) with caching
    with st.spinner("Preparing backend (Phase‑1 + training)…"):
        prep = _prepare(str(csv_path), process_col, id_col, composition_prefix, embed_model, birch_threshold, agglo_clusters)

    targets = prep["targets"]
    if not targets:
        st.error("No numeric target columns detected. Please include columns like UTS, YS, Elongation, Hardness, etc.")
        st.stop()

    st.success("Backend ready. Enter desired properties below to get alloy recommendations.")

    st.markdown("### 🎯 Desired properties")
    cols = st.columns(min(4, len(targets)))
    target_vals = {}
    for i, name in enumerate(targets):
        col = cols[i % len(cols)]
        default_val = float(df[name].dropna().median()) if name in df.columns else 0.0
        target_vals[name] = col.number_input(f"Target {name}", value=default_val)

    st.markdown("#### Optional filters")
    grade_col_guess = next((g for g in ["grade", "alloy", "material", "spec", "standard"] if g in df.columns), None)
    grade_col = st.selectbox("Grade/standard column (optional)", options=["(none)"] + list(df.columns), index=(["(none)"] + list(df.columns)).index(grade_col_guess) if grade_col_guess else 0)
    include_kw = st.text_input(f"Include keyword in `{process_col}` (optional)")
    exclude_kw = st.text_input(f"Exclude keyword in `{process_col}` (optional)")
    topk = st.slider("Top K", 5, 100, 15, 5)

    if st.button("🎯 Recommend alloys"):
        kw_col = None if grade_col == "(none)" else process_col
        recs = recommend_alloys(
            df=df,
            property_cols=targets,
            targets=target_vals,
            top_k=topk,
            include_keyword_col=process_col,
            include_keyword=include_kw or None,
            exclude_keyword=exclude_kw or None,
        )
        if grade_col != "(none)" and not recs.empty:
            # show grade first if available
            cols = [grade_col] + [c for c in recs.columns if c != grade_col]
            recs = recs[cols]
        if recs.empty:
            st.warning("No matching recommendations. Try relaxing filters.")
        else:
            st.success(f"Top {len(recs)} recommendations")
            st.dataframe(recs)
            st.download_button("⬇️ Download CSV", recs.to_csv(index=False).encode("utf-8"), "recommendations.csv", "text/csv")

    st.markdown("---")
    with st.expander("🔎 Advanced search (semantic text)"):
        model_choice = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"], index=0, key="search_model")
        query = st.text_input("Find similar process text (e.g., 'quenched and tempered 0.4% C')", key="search_query")
        k2 = st.slider("Top K (search)", 5, 100, 25, 5, key="search_topk")
        if st.button("🔎 Run search"):
            hits = semantic_search(df, text_col=process_col, query=query, model_name=model_choice, top_k=k2)
            if grade_col != "(none)" and not hits.empty and grade_col in hits.columns:
                cols = [grade_col] + [c for c in hits.columns if c != grade_col]
                hits = hits[cols]
            st.dataframe(hits)
            if not hits.empty:
                st.download_button("⬇️ Download search results", hits.to_csv(index=False).encode("utf-8"), "search_results.csv", "text/csv")
