import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Feature 1 = Email length
# Feature 2 = Number of urgent words

X = np.array([
    [50, 1],
    [55, 2],
    [60, 1],
    [65, 2],
    [70, 1],
    [75, 2],
    [80, 1],
    [85, 2],
    [90, 1],
    [95, 2],
    [100, 1],
    [105, 2],
    [110, 2],
    [115, 2],

    [120, 5],
    [125, 6],
    [130, 5],
    [135, 7],
    [140, 6],
    [145, 7],
    [150, 6],
    [155, 8],
    [160, 7],
    [165, 8],
    [170, 7],
    [175, 8],
    [180, 7]
])

y = np.array([
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,

    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1
])

model = SVC(kernel="linear")

model.fit(X, y)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
confusionmatrix = confusion_matrix(y, y_pred)

print("Actual values:", y)
print("\nPredicted values: ", y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusionmatrix)

w = model.coef_[0]
b = model.intercept_[0]

print("\nWeights: ", w)
print("\nBias: ", b)

print("\nSupport Vectors:")
print(model.support_vectors_)

print("\nNumber of Support Vectors:")
print(model.n_support_)

x1 = np.linspace( X[:, 0].min() - 10, X[:, 0].max() + 10, 100)

# Decision Boundary: w1*x1 + w2*x2 + b = 0
x2 = -(w[0] * x1 + b) / w[1]

# Positive Margin: w1*x1 + w2*x2 + b = 1
positive_margin = -(w[0] * x1 + b - 1) / w[1]

# Negative Margin: w1*x1 + w2*x2 + b = -1
negative_margin = -(w[0] * x1 + b + 1) / w[1]

plt.figure(figsize=(10, 7))

# Not Spam points
plt.scatter( X[y == 0, 0], X[y == 0, 1], label="Not Spam", s=70)

# Spam points
plt.scatter( X[y == 1, 0], X[y == 1, 1], label="Spam", s=70)

# Decision boundary
plt.plot( x1, x2, label="Decision Boundary", linewidth=2)

# Positive margin
plt.plot( x1, positive_margin, "--", label="Positive Margin")

# Negative margin
plt.plot( x1, negative_margin, "--", label="Negative Margin")

plt.scatter( model.support_vectors_[:, 0], model.support_vectors_[:, 1],
    s=200, facecolors="none", edgecolors="black", linewidths=2, label="Support Vectors")

plt.xlabel("Email Length")
plt.ylabel("Urgent Word Count")
plt.title("SVM Spam Detection - Decision Boundary and Margins")
plt.legend()
plt.grid(True)
plt.show()