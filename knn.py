import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def predict(X_train, y_train, X_test, k=5):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)

    predictions = []

    for test_point in X_test:
        distances = np.sum(
            (X_train - test_point) ** 2,
            axis=1
        )

        nearest_indices = np.argpartition(
            distances,
            kth=k - 1
        )[:k]

        nearest_labels = y_train[nearest_indices]

        labels, counts = np.unique(
            nearest_labels,
            return_counts=True
        )

        prediction = labels[np.argmax(counts)]
        predictions.append(prediction)

    return np.asarray(predictions)

def knn(X_train, y_train, X_test, y_test, k=5):
    predictions = predict(
        X_train,
        y_train,
        X_test,
        k
    )

    return np.mean(predictions == np.asarray(y_test))

df = pd.read_csv('data/df_final.csv')

y = df["movimiento"].values

X = df.drop(
    columns=["movimiento", "repeticion_id"]
).values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

accuracy = knn(
    X_train,
    y_train,
    X_test,
    y_test,
    k=5
)

predictions = predict(X_train, y_train, X_test, k=5)

print("Accuracy:", accuracy)
print("Predictions:", predictions[0:10])
print("Real:", y_test[0:10])