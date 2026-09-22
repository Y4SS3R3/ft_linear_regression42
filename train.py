import json
import matplotlib.pyplot as plt
from utils import load_dataset, normalize, denormalize, estimatePrice

df = load_dataset()

df['km_normalized'] = normalize(df['km'], df['km'].min(), df['km'].max())
df['price_normalized'] = normalize(df['price'], df['price'].min(), df['price'].max())

theta0 = 0.0
theta1 = 0.0
learning_rate = 0.5
epochs = 10000

for e in range(epochs):
    error = estimatePrice(df['km_normalized'], theta0, theta1) - df['price_normalized']
    theta0 -= learning_rate * error.mean()
    theta1 -= learning_rate * (error * df['km_normalized']).mean()


min_km = float(df['km'].min())
max_km = float(df['km'].max())

min_price = float(df['price'].min())
max_price = float(df['price'].max())

data = {
    "t0": theta0,
    "t1": theta1,
    "min_km": min_km,
    "max_km": max_km,
    "min_price": min_price,
    "max_price": max_price
}

with open("thetas.json", 'w') as file:
    json.dump(data, file, indent=4)


plt.title('Car Price vs. Mileage')
plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.scatter(df['km'], df['price'], color='purple', label='Actual Data')

predicted_prices = denormalize(estimatePrice(df['km_normalized'], theta0, theta1), min_price, max_price)
plt.plot(df['km'], predicted_prices, color='orange', label='Linear Regression Line')

plt.legend()
plt.savefig("regression_graph.png")
