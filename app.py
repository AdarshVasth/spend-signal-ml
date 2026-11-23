from pathlib import Path

import pandas as pd
import streamlit as st

from spend_signal import InputError, analyze_transactions

st.set_page_config(page_title="SpendSignal", page_icon="◈", layout="wide")
st.title("SpendSignal")
st.caption("Find transactions that stand out in your own data. Flags are review prompts, not fraud verdicts.")

with st.sidebar:
    st.header("Analyze a file")
    uploaded = st.file_uploader("Upload a transaction CSV", type="csv", max_upload_size=10)
    contamination = st.slider("Expected unusual share", min_value=1, max_value=20, value=5, step=1,
                              help="Controls how many rows the model labels as unusual. It is not a fraud probability.") / 100
    st.markdown("Required columns: `timestamp`, `amount`, `merchant`, `category`, `channel`.")
    st.download_button("Download example CSV", Path("data/sample_transactions.csv").read_bytes(),
                       file_name="sample_transactions.csv", mime="text/csv")

try:
    if uploaded is None:
        frame = pd.read_csv("data/sample_transactions.csv")
        st.info("Showing the included synthetic example. Upload your own CSV from the sidebar to analyze it.")
    else:
        frame = pd.read_csv(uploaded)
    analysis = analyze_transactions(frame, contamination=contamination)
except (InputError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(str(exc))
    st.stop()

result = analysis.results
flagged = result[result["is_unusual"]]
left, middle, right = st.columns(3)
left.metric("Transactions", f"{len(result):,}")
middle.metric("Flagged for review", f"{analysis.flagged_count:,}")
right.metric("Flag rate", f"{analysis.flagged_count / len(result):.1%}")

st.subheader("Amount pattern")
chart = result.copy()
chart["row"] = range(1, len(chart) + 1)
st.scatter_chart(chart, x="row", y="amount", color="is_unusual", height=320)
st.caption("Rows are sorted by anomaly score. The model learns patterns in this file; it does not compare against a bank or outside dataset.")

st.subheader("Review queue")
display_columns = ["timestamp", "amount", "merchant", "category", "channel", "anomaly_score", "review_note"]
st.dataframe(flagged[display_columns], use_container_width=True, hide_index=True)
st.download_button("Download analyzed CSV", result.to_csv(index=False).encode("utf-8"),
                   file_name="spendsignal_results.csv", mime="text/csv")

with st.expander("How it works and what it cannot tell you"):
    st.markdown("""
    SpendSignal uses **Isolation Forest**, an unsupervised outlier-detection algorithm. It uses amount,
    time, merchant, category, and channel patterns from the uploaded file. Lower anomaly scores mean
    the row was easier for the model to isolate.

    A flagged row is **not** evidence of fraud, and an unflagged row is not proof a transaction is safe.
    The result depends on the rows you provide and the expected unusual share. Review records yourself.
    The app processes the CSV in the current app session and does not save it to a database.
    """)
