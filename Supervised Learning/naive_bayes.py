import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Machine-Learning/Supervised Learning/spam_email_svm_numerical.csv")

X = df[
    [
        "Email_Length",
        "Number_of_Links",
        "Number_of_Attachments",
        "Money_Word_Count",
        "Urgent_Word_Count",
        "Sender_Reputation"
    ]
].values.tolist()

y = df["Spam"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.2, random_state=20)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusionmatrix = confusion_matrix(y_test, y_pred)

print("Actual values: ", y_test)
print("\nPredicted values: ", y_pred)

print("\nAccuracy: ", accuracy)
print("\nConfusion Matrix:")
print(confusionmatrix)