import streamlit as st
import joblib, pathlib
from typing import Optional
from rs_gpt import RSInterventionGPT

st.set_page_config(page_title="Road Safety Intervention GPT", layout="wide")

# Use Streamlit native components for layout to respect theme (dark/light)

st.title("🚦 Road Safety Intervention GPT")
st.caption("Hybrid scoring with tunable weights and feedback. Load the model, enter a query and get ranked interventions.")

with st.sidebar:
    st.header("Model & Weights")
    model_dir = st.text_input("Model directory", value="models")
    load = st.button("Load model")

    st.markdown("---")
    st.subheader("Optional live weights")
    w_tfidf = st.slider("TF-IDF weight", 0.0, 1.0, 0.7, 0.01)
    w_embed = st.slider("Embeddings weight", 0.0, 1.0, 0.2, 0.01)
    w_rules = st.slider("Rules weight", 0.0, 1.0, 0.1, 0.01)
    w_feedback = st.slider("Feedback weight", 0.0, 1.0, 0.05, 0.01)

if "model" not in st.session_state:
    st.session_state.model = None

if load:
    try:
        m = RSInterventionGPT.load(model_dir)
        # apply live weights if config supports them
        if hasattr(m.cfg, "w_tfidf"):
            m.cfg.w_tfidf = w_tfidf
            m.cfg.w_embed = w_embed
            m.cfg.w_rules = w_rules
            m.cfg.w_feedback = w_feedback
        st.session_state.model = m
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

query = st.text_area("Describe the issue", value="faded stop sign near intersection at night involving two-wheelers", height=120)
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    topk = st.number_input("Top K", 1, 30, 5)
with col2:
    category = st.text_input("Filter category (optional)")
with col3:
    type_filter = st.text_input("Filter type (optional)")

if st.button("Get recommendations"):
    if st.session_state.model is None:
        st.warning("Please load a trained model from the sidebar first.")
    else:
        # update weights if supported
        if hasattr(st.session_state.model.cfg, "w_tfidf"):
            st.session_state.model.cfg.w_tfidf = w_tfidf
            st.session_state.model.cfg.w_embed = w_embed
            st.session_state.model.cfg.w_rules = w_rules
            st.session_state.model.cfg.w_feedback = w_feedback

        res = st.session_state.model.recommend(query, top_k=topk, category=category or None, type_filter=type_filter or None)
        if res.empty:
            st.info("No matches found for your query.")
        else:
            # show download and quick copy
            st.download_button("📥 Download results (CSV)", res.to_csv(index=False), "recommendations.csv", "text/csv")

            for i, row in res.iterrows():
                # format score as percentage (numeric)
                score_pct = float(row['score']) * 100.0
                # layout two columns: content | score
                left, right = st.columns([8, 1])
                with left:
                    st.subheader(f"Recommendation #{int(row['rank'])}")
                    st.caption(f"Category: {row['category']} | Type: {row['type']}")
                    st.markdown("---")
                    st.markdown(f"**Problem:** {row['problem']}")
                    st.markdown(f"**Intervention:** {row['explanation']}")
                    with st.expander("📚 IRC Reference", expanded=False):
                        st.write(f"Code: {row['reference_code']}")
                        st.write(f"Clause: {row['reference_clause']}")
                    # feedback button
                    if st.button(f"👍 Accept Rank {int(row['rank'])}", key=f"fb_{i}"):
                        if hasattr(st.session_state.model, 'record_feedback'):
                            try:
                                st.session_state.model.record_feedback(row, positive=True)
                                joblib.dump(st.session_state.model.feedback_counts, pathlib.Path(model_dir) / "feedback_counts.joblib")
                                st.success("Recorded feedback.")
                            except Exception:
                                st.error("Could not save feedback.")
                with right:
                    # show as metric with percentage
                    st.metric(label="Relevance", value=f"{score_pct:.1f}%")
                st.divider()
