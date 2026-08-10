from sklearn.linear_model import Perceptron

work = [
    [2, 2, 0, 1],
    [5, 5, 1, 0],
    [1, 1, 0, 1],
    [7, 6, 1, 1],
    [3, 2, 1, 1],
    [2, 4, 1, 0],
    [2, 3, 0, 0],
    [8, 7, 1, 0]
]

y = [1, 0, 1, 1, 1, 0, 1, 0]

# Create single-layer perceptron
model = Perceptron()

# Train
model.fit(work, y)

# Test training data
for w in work:
    print(w, model.predict([w])[0])


# Get new order information
dist = float(input("Enter Distance: "))
item = int(input("Enter no. of items: "))
peak = int(input("Enter peak time (0/1): "))
rider = int(input("Rider avail (0/1): "))

# New input
inputdata = [[dist, item, peak, rider]]

# Prediction
result = model.predict(inputdata)[0]

if result == 1:
    print("Order Accepted")
else:
    print("Order Rejected")