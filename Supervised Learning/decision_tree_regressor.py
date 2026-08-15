import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv("Machine-Learning/Supervised Learning/decision_tree_regression_houses.csv")

X = df[
    [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Age_years"
    ]
].values.tolist()

y = df["Price"].values.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)

# Create Decision Tree Regression model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Predict prices using the TEST FEATURES
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

#Visualize the data
plot_tree(model, feature_names=[
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_years"
], filled=True)

plt.show()