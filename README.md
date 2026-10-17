# Decision Tree Explorer

An original decision-tree lesson using scikit-learn's built-in Iris dataset.
Adjust the maximum depth and inspect readable learned rules and hold-out score.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

This small sample is for learning and does not guarantee results on other data.
Run tests with `python -m unittest discover -s tests -v`.
