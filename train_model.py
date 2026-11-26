
# train_model.py
# Train and save a simple KNN model using Python 3.14

import csv
import random
import math
import pickle
import urllib.request

print("Loading Iris Dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

response = urllib.request.urlopen(url, timeout=30)
lines = response.read().decode("utf-8").splitlines()

X = []
y = []

for line in lines:
    if not line.strip():
        continue

    row = next(csv.reader([line]))
    X.append([float(v) for v in row[:4]])
    y.append(row[4])

print("Dataset Loaded Successfully!")
print("Total Records:", len(X))

# Split dataset into training and testing
data = list(zip(X, y))

random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.8)

train_data = data[:split]
test_data = data[split:]

X_train = [item[0] for item in train_data]
y_train = [item[1] for item in train_data]

X_test = [item[0] for item in test_data]
y_test = [item[1] for item in test_data]

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# KNN prediction function
def predict_one(row, training_X, training_y, k=5):

    distances = []

    for features, label in zip(training_X, training_y):

        distance = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(row, features)
            )
        )

        distances.append((distance, label))

    distances.sort(key=lambda item: item[0])

    nearest = distances[:k]

    votes = {}

    for distance, label in nearest:
        votes[label] = votes.get(label, 0) + 1

    prediction = max(votes, key=votes.get)

    confidence = votes[prediction] / k

    return prediction, confidence


# Evaluate model
correct = 0

for features, actual in zip(X_test, y_test):

    predicted, confidence = predict_one(
        features, X_train, y_train
    )

    if predicted == actual:
        correct += 1

accuracy = correct / len(y_test) * 100

print("\nModel Training Completed!")
print("Model Accuracy:", round(accuracy, 2), "%")


# Save model with Pickle
model = {
    "X_train": X_train,
    "y_train": y_train,
    "classes": sorted(set(y)),
    "accuracy": accuracy,
    "feature_names": [
        "Sepal length",
        "Sepal width",
        "Petal length",
        "Petal width"
    ]
}

with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully!")
print("File Name: iris_model.pkl")
