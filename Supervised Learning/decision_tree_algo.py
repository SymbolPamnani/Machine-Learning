import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv("Machine-Learning/Supervised Learning/decision_tree_orders.csv")

X = df[
    [
        "Distance_km",
        "Items",
        "Rider_Available",
        "Peak_Time"
    ]
].values.tolist()

y = df["Order_Accepted"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy= accuracy_score(y_pred, y_test)

cofusionmatrix= confusion_matrix(y_pred, y_test)

print("Predicted value: ", y_pred)
print("Accuracy score: ", accuracy)
print("Confusion Matrix: ", cofusionmatrix)

plot_tree(model, feature_names=[
    "Distance_km",
    "Items",
    "Rider_Available",
    "Peak_Time"
], class_names=["Rejected", "Accepted"], filled=True)

plt.show()