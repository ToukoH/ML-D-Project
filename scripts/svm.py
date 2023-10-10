from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd

df = pd.read_csv("../data/diabetes.csv")

X = df.drop("Outcome", axis="columns")
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)

svm = SVC(kernel="linear",
          C=1)

svm.fit(X_train, y_train)

y_train_pred = svm.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)
training_error = 1 - training_accuracy

print(f"Training Error: {training_error:.4f}")
print("____________________________")

y_val_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

scores = cross_val_score(svm, X_train,
                         y_train, cv=5, scoring='accuracy')

print("____________________________")
print(f"Fold Accuracies: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f}")
print("____________________________")

svm.fit(X_train, y_train)

y_test_pred = svm.predict(X_test)

accuracy_score = accuracy_score(y_test, y_test_pred)
print(f"Accuracy Score: {accuracy_score:.4f}")
