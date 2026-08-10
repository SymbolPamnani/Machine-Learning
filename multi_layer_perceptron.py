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


# -------------------------
# Perceptron
# -------------------------

from sklearn.linear_model import Perceptron

model = Perceptron()
model.fit(work, y)

print("Perceptron Predictions:")

for w in work:
    print(w, model.predict([w])[0])


# -------------------------
# Neural Network
# -------------------------

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

workscale = scaler.fit_transform(work)

model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='logistic',
    max_iter=1000,
    random_state=42
)

model.fit(workscale, y)


print("\nNeural Network Predictions:")

for w in workscale:
    print(w, model.predict([w])[0])


# -------------------------
# User Input
# -------------------------

dist = float(input("\nEnter Distance: "))
item = int(input("Enter no. of items: "))
peak = int(input("Enter peak time (0/1): "))
rider = int(input("Rider avail (0/1): "))


# Original input
inputdata = [[dist, item, peak, rider]]


# Scale using the SAME scaler used during training
inputscaled = scaler.transform(inputdata)


# Prediction
result = model.predict(inputscaled)[0]


if result == 1:
    print("Order Accepted")
else:
    print("Order Rejected")