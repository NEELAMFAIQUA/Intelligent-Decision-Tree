import streamlit as st
import pandas as pd

from tree_demo import train_tree

st.set_page_config(page_title="Decision Tree Explorer", page_icon="🌳")
st.title("Decision Tree Explorer")
st.caption("Upload a tabular CSV, select its label column, then explore the learned tree.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"],
        help="Maximum upload size: 5 MB. The file is processed in this app session.",
    )
    st.caption("Need an example? Use the bundled sample dataset below.")
    use_sample = st.checkbox("Use bundled weather-planning sample", value=uploaded_file is None)
    depth = st.slider("Maximum tree depth", 1, 12, 3)
    seed = st.number_input("Random seed", min_value=0, max_value=10000, value=7)

if uploaded_file and not use_sample:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("CSV upload exceeds the 5 MB limit.")
        st.stop()
    try:
        dataset = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"Could not read this CSV: {error}")
        st.stop()
    dataset_name = uploaded_file.name
else:
    dataset = pd.read_csv("data/sample_decisions.csv")
    dataset_name = "Bundled weather-planning sample"

if dataset.empty or len(dataset.columns) < 2:
    st.error("Dataset must contain rows and at least two columns.")
    st.stop()

st.subheader(dataset_name)
st.dataframe(dataset.head(100), use_container_width=True)
st.caption(f"{len(dataset):,} rows · {len(dataset.columns)} columns")
target = st.selectbox(
    "Target / label column",
    list(dataset.columns),
    index=len(dataset.columns) - 1,
)

try:
    result = train_tree(dataset, target, depth, int(seed))
except ValueError as error:
    st.warning(str(error))
    st.stop()

left, right, extra = st.columns(3)
left.metric("Hold-out accuracy", f"{result['accuracy']:.1%}")
right.metric("Learned depth / leaves", f"{result['depth']} / {result['leaves']}")
extra.metric("Test rows", result["test_count"])
st.caption("Accuracy is measured on a stratified hold-out split of this dataset.")

st.subheader("Confusion matrix")
st.caption("Rows are actual labels; columns are predicted labels. Class order: " + ", ".join(result["classes"]))
st.dataframe(
    pd.DataFrame(
        result["confusion_matrix"],
        index=[f"actual {label}" for label in result["classes"]],
        columns=[f"predicted {label}" for label in result["classes"]],
    ),
    use_container_width=True,
)

with st.expander("Readable decision rules"):
    st.code(result["rules"], language="text")

st.info(
    "This tool is for learning from tabular data. A single train/test score can "
    "be misleading, especially on small or imbalanced datasets. Do not use it "
    "alone to make consequential decisions."
)
