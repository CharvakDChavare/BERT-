import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BERT Sentiment Analyser",
    page_icon="🎯",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("🎯 BERT Sentiment Analyser")
st.markdown("Powered by fine-tuned `bert-base-uncased` · Trained on Yelp & Amazon reviews")
st.divider()

# ── API URL ───────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Single prediction ─────────────────────────────────────────
st.subheader("Single Review")
text = st.text_area(
    "Enter a review:",
    placeholder="e.g. This product is absolutely amazing!",
    height=120
)

if st.button("Analyse", use_container_width=True):
    if text.strip() == "":
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text}
                )
                result = response.json()

                label      = result["label"]
                confidence = result["confidence"]

                # Display result
                if label == "Positive":
                    st.success(f"Positive  —  {confidence:.2%} confidence")
                else:
                    st.error(f"Negative  —  {confidence:.2%} confidence")

                # Confidence bar
                st.progress(confidence)

            except Exception as e:
                st.error(f"API error: {e}")

st.divider()

# ── Batch prediction ──────────────────────────────────────────
st.subheader("Batch Reviews")
st.caption("Enter one review per line")

batch_text = st.text_area(
    "Enter multiple reviews:",
    placeholder="Great product, love it!\nTerrible quality, waste of money.\nIt was okay, nothing special.",
    height=160,
    key="batch"
)

if st.button("Analyse All", use_container_width=True):
    lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
    if not lines:
        st.warning("Please enter at least one review.")
    else:
        with st.spinner("Analysing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    json={"texts": lines}
                )
                results = response.json()["results"]

                st.markdown("#### Results")
                for r in results:
                    label      = r["label"]
                    confidence = r["confidence"]
                    text_short = r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"]

                    col1, col2, col3 = st.columns([5, 2, 2])
                    with col1:
                        st.write(text_short)
                    with col2:
                        if label == "Positive":
                            st.success(label)
                        else:
                            st.error(label)
                    with col3:
                        st.write(f"{confidence:.2%}")

            except Exception as e:
                st.error(f"API error: {e}")

st.divider()

# ── Model info ────────────────────────────────────────────────
with st.expander("Model details"):
    st.markdown("""
    | Property | Value |
    |---|---|
    | Base model | bert-base-uncased |
    | Training data | Yelp Polarity (1000 samples) |
    | Test accuracy | 92% (Yelp), 88% (Amazon) |
    | F1 score | 0.921 |
    | API | FastAPI on port 8000 |
    | Deployment | Docker container |
    """)