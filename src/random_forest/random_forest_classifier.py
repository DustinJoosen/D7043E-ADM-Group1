import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier

CLASS_ORDER = ["N", "S", "V", "F"]
DATA_FOLDER = "../data/"


def load_npz_X_y(file_name: str):
    dataset = numpy.load(file_name)
    return dataset["X"], dataset["y"]


def evaluate(classifier, X_true, y_true, title=""):
    y_pred = classifier.predict(X_true)

    print(classification_report(y_true, y_pred, labels=CLASS_ORDER))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

    conf_counts = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    conf_norm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER, normalize="true")

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(conf_counts, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[0]).set_title(f"{title} counts")
    sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[1]).set_title(f"{title} normalized")

    for ax in axes:
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    plt.show()


X_train, y_train = load_npz_X_y(f"{DATA_FOLDER}/train.npz")
X_val, y_val = load_npz_X_y(f"{DATA_FOLDER}/val.npz")
X_test, y_test = load_npz_X_y(f"{DATA_FOLDER}/test.npz")

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Change the X_test and y_test -> X_val and y_val if ya want.
evaluate(classifier, X_test, y_test, "Test")

with open("random_forest_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
