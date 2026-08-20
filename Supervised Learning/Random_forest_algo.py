import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
df =  pd.read_csv("Machine-Learning/Supervised Learning/spam_email_random_forest.csv")

X = df[
    [
        "Email_Length",
        "Contains_Link",
        "Contains_Attachment",
        "Contains_Money_Words",
        "Contains_Urgent_Words",
        "Sender_Known"
    ]
].values.tolist()

y = df["Spam"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy= accuracy_score(y_pred, y_test)

confusionmatrix= confusion_matrix(y_pred, y_test)

print("\nActual value: ", y_test)
print("\nPredicted value: ", y_pred)
print("\nAccuracy: ", accuracy)
print("\nConfusion matrix: ", confusionmatrix)

plot_tree(
    model.estimators_[0],
    feature_names=[
        "Email_Length",
        "Contains_Link",
        "Contains_Attachment",
        "Contains_Money_Words",
        "Contains_Urgent_Words",
        "Sender_Known"
    ],
    class_names=["Not Spam", "Spam"],
    filled=True
)

plt.show()