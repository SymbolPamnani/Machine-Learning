import numpy as np
import matplotlib.pyplot as plt

# Two features: x1 and x2
X = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [7, 7],
    [8, 7],
    [8, 8],
    [9, 8]
])

# Two classes
# -1 = negative class
# +1 = positive class
y = np.array([
    -1,
    -1,
    -1,
    -1,
     1,
     1,
     1,
     1
])

from sklearn.svm import SVC
model = SVC(kernel="linear")
model.fit(X, y)
y_pred = model.predict(X)

print("Actual:", y)
print("Predicted:", y_pred)

print("Support Vectors:")
print(model.support_vectors_)

print("Support Vector Indices:")
print(model.support_)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

w = model.coef_[0]
b = model.intercept_[0]

print("w =", w)
print("b =", b)

x1 = np.linspace(0, 10, 100)

x2 = -(w[0] * x1 + b) / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.plot(x1, x2)

plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="black"
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("SVM Decision Boundary")

plt.show()