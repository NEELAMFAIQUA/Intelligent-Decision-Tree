"""Reproducible decision-tree lesson using scikit-learn's Iris dataset."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def train_tree(max_depth=3, seed=7):
    if not 1 <= max_depth <= 8:
        raise ValueError("max_depth must be between 1 and 8")
    data = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=seed,
        stratify=data.target,
    )
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    model.fit(train_x, train_y)
    return {
        "accuracy": float(accuracy_score(test_y, model.predict(test_x))),
        "rules": export_text(model, feature_names=list(data.feature_names)),
        "classes": list(data.target_names),
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
    }
