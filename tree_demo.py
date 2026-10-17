"""Decision-tree training helpers for user-provided tabular CSV datasets."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def train_tree(
    data: pd.DataFrame,
    target_column: str,
    max_depth: int = 3,
    seed: int = 7,
) -> dict:
    """Fit a reproducible classifier and return its rules and test metrics."""
    if not 1 <= max_depth <= 12:
        raise ValueError("max_depth must be between 1 and 12")
    if target_column not in data.columns:
        raise ValueError("Choose a target column that exists in the CSV")
    if len(data.columns) < 2:
        raise ValueError("CSV must include at least one feature and one target column")

    cleaned = data.dropna(subset=[target_column]).copy()
    if len(cleaned) < 10:
        raise ValueError("CSV needs at least 10 rows with a target value")
    targets = cleaned[target_column]
    class_counts = targets.value_counts()
    if len(class_counts) < 2:
        raise ValueError("The target column must contain at least two classes")
    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified split")

    raw_features = cleaned.drop(columns=[target_column])
    features = pd.get_dummies(raw_features, dummy_na=True, dtype=float)
    if features.empty:
        raise ValueError("CSV must contain at least one usable feature column")
    if len(features.columns) > 250:
        raise ValueError("CSV has too many encoded feature columns (maximum 250)")

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=seed,
        stratify=targets,
    )
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    classes = model.classes_.tolist()
    return {
        "accuracy": float(accuracy_score(test_y, prediction)),
        "confusion_matrix": confusion_matrix(test_y, prediction, labels=classes).tolist(),
        "classes": [str(label) for label in classes],
        "rules": export_text(model, feature_names=list(features.columns)),
        "feature_names": list(features.columns),
        "row_count": len(cleaned),
        "test_count": len(test_y),
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
    }
