import pandas as pd

df = pd.read_csv("decision_tree_orders.csv")

print(df.head())

X = df[
    [
        "Distance_km",
        "Items",
        "Rider_Available",
        "Peak_Time"
    ]
].values.tolist()

y = df["Order_Accepted"].tolist()


# GINI IMPURITY
def gini(labels):

    # Total number of observations
    total = len(labels)

    # Count how many observations belong to class 0
    count_0 = labels.count(0)

    # Count how many observations belong to class 1
    count_1 = labels.count(1)

    # Calculate probability of class 0
    p0 = count_0 / total

    # Calculate probability of class 1
    p1 = count_1 / total

    # Gini Impurity formula:
    # Gini = 1 - (p0² + p1²)
    gini_value = 1 - (p0 ** 2 + p1 ** 2)

    return gini_value

# Test our Gini function
print("\nGini Test:")

print(
    gini([1, 1, 1, 0, 0, 0])
)

#  SPLIT THE DATA

def split_data(X, y, feature_index, threshold):

    # Lists for the LEFT side of the split
    left_X = []
    left_y = []

    # Lists for the RIGHT side of the split
    right_X = []
    right_y = []

    # Go through every row in the dataset
    for i in range(len(X)):

        # If the feature value is less than or equal to the threshold, put the row on the LEFT
        if X[i][feature_index] <= threshold:

            left_X.append(X[i])
            left_y.append(y[i])

        # Otherwise, put the row on the RIGHT
        else:

            right_X.append(X[i])
            right_y.append(y[i])

    return left_X, left_y, right_X, right_y

# CALCULATE WEIGHTED GINI FOR A SPLIT

def calculate_split_gini(X, y, feature_index, threshold):

    # First, split the dataset
    left_X, left_y, right_X, right_y = split_data(
        X,
        y,
        feature_index,
        threshold
    )

    # If one side contains no data, this is not a useful split
    if len(left_y) == 0 or len(right_y) == 0:
        return float("inf")

    # Calculate Gini impurity for the LEFT group
    left_gini = gini(left_y)

    # Calculate Gini impurity for the RIGHT group
    right_gini = gini(right_y)

    # Total number of observations
    total = len(y)

    # Weight of the LEFT group
    left_weight = len(left_y) / total

    # Weight of the RIGHT group
    right_weight = len(right_y) / total

    # Weighted Gini formula:
    # Split Gini =
    #     (Left size / Total size) * Left Gini
    #   + (Right size / Total size) * Right Gini
    
    split_gini = (
        left_weight * left_gini
        + right_weight * right_gini
    )

    return split_gini

# FIND THE BEST SPLIT

def find_best_split(X, y):

    # Start with the worst possible Gini value
    # We want to find something smaller than this.
    best_gini = float("inf")

    # We will store the feature that produces the best split
    best_feature = None

    # We will store the threshold that produces the best split
    best_threshold = None
    number_of_features = len(X[0])  # Number of features

    # Loop through every feature
    for feature_index in range(number_of_features):
        # Get all values for this feature
        feature_values = []

        for row in X:
            feature_values.append(row[feature_index])
        unique_values = sorted(set(feature_values))    # Remove duplicate values and sort them

        for i in range(len(unique_values) - 1):
            current_value = unique_values[i]
            next_value = unique_values[i + 1]
            threshold = (current_value + next_value) / 2   # Calculate midpoint

            # Calculate the Gini score for this split
            current_gini = calculate_split_gini(
                X,
                y,
                feature_index,
                threshold
            )

            # If this split has a lower Gini, it is currently our best split
            if current_gini < best_gini:

                best_gini = current_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gini

# FIND THE BEST FIRST SPLIT
best_feature, best_threshold, best_gini = find_best_split(X, y)

# DISPLAY THE RESULT
feature_names = [
    "Distance_km",
    "Items",
    "Rider_Available",
    "Peak_Time"
]

print("\nBEST FIRST SPLIT")
print("Best Feature:", feature_names[best_feature])
print("Best Threshold:", best_threshold)
print("Best Gini:", best_gini)

# SHOW THE ACTUAL GROUPS CREATED BY THE BEST SPLIT

left_X, left_y, right_X, right_y = split_data(
    X,
    y,
    best_feature,
    best_threshold
)

print("\nLEFT GROUP")

print("Number of observations:", len(left_y))
print("Accepted:", left_y.count(1))
print("Rejected:", left_y.count(0))
print("Gini:", gini(left_y))


print("\nRIGHT GROUP")

print("Number of observations:", len(right_y))
print("Accepted:", right_y.count(1))
print("Rejected:", right_y.count(0))
print("Gini:", gini(right_y))