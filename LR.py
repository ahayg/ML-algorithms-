import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

class LogisticRegressionScratch:
    """Binary Logistic Regression via gradient descent:

    sigmoid(z)  = 1 / (1 + e^-z)
    y^          = sigmoid(X @ w + b)
    Loss (BCE)  = -[y log(y^) + (1-y) log(1-y^)]
    ∂Loss/∂w    = (1/m) * Xᵀ (ŷ - y)
    ∂Loss/∂b    = (1/m) * sum(ŷ - y)
    """

    def __init__(self, lr=0.01, epochs=1000, threshold=0.5):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y, y_hat):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for epoch in range(self.epochs):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            dw = (1 / m) * X.T @ (y_hat - y)
            db = (1 / m) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                loss = self._binary_cross_entropy(y, y_hat)
                self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def run_demo():
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=6,
        n_redundant=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("=" * 55)
    print(" FROM SCRATCH LOGISTIC REGRESSION")
    print("=" * 55)

    scratch_model = LogisticRegressionScratch(lr=0.1, epochs=1000)
    scratch_model.fit(X_train_sc, y_train)

    y_pred_scratch = scratch_model.predict(X_test_sc)
    y_proba_scratch = scratch_model.predict_proba(X_test_sc)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred_scratch):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred_scratch):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred_scratch):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred_scratch):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba_scratch):.4f}")

    print("\n" + "=" * 55)
    print(" SKLEARN LOGISTIC REGRESSION")
    print("=" * 55)

    sk_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    sk_model.fit(X_train_sc, y_train)

    y_pred_sk = sk_model.predict(X_test_sc)
    y_proba_sk = sk_model.predict_proba(X_test_sc)[:, 1]

    print(f"Accuracy  : {accuracy_score(y_test, y_pred_sk):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred_sk):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred_sk):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred_sk):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba_sk):.4f}")

    print("\nFull Classification Report")
    print(classification_report(y_test, y_pred_sk))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(0, 1000, 100), scratch_model.loss_history, marker='o', color='steelblue')
    axes[0].set_title("Training Loss (From Scratch)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross Entropy")
    axes[0].grid(True, alpha=0.3)

    cm = confusion_matrix(y_test, y_pred_sk)
    axes[1].imshow(cm, cmap='Blues')
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cm[i, j], ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black',
                         fontsize=16)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])

    plt.tight_layout()
    plt.savefig("logistic_regression_results.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to logistic_regression_results.png")


if __name__ == "__main__":
    run_demo()