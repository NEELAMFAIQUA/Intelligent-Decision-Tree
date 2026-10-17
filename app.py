import streamlit as st

from tree_demo import train_tree

st.set_page_config(page_title="Decision Tree Explorer", page_icon="🌳")
st.title("Decision Tree Explorer")
st.caption("Fit a small decision tree and read the rules it learned.")
depth = st.slider("Maximum tree depth", 1, 8, 3)
result = train_tree(depth)
left, right = st.columns(2)
left.metric("Hold-out accuracy", f"{result['accuracy']:.1%}")
right.metric("Learned depth / leaves", f"{result['depth']} / {result['leaves']}")
st.subheader("Readable decision rules")
st.code(result["rules"], language="text")
st.caption("Output classes: " + ", ".join(result["classes"]))
st.info(
    "The Iris dataset is a compact teaching example. A single hold-out score "
    "does not guarantee performance on other datasets."
)
