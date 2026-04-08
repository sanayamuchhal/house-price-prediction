import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# load dataset
df = pd.read_csv("data.csv")

# clean column names
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)

X = pd.get_dummies(df[['size', 'bedrooms', 'city']], drop_first=True)

Y = df['price']

# split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# train model
model = LinearRegression()
model.fit(X_train, Y_train)

print("Model trained")

# predictions
Y_pred = model.predict(X_test)

# compare
compare = pd.DataFrame({
    'Actual': Y_test.astype(int),'Predicted': Y_pred.astype(int)})
print("\nActual vs Prediction:\n", compare.head())

# evaluation
print("\nR2 Score:", r2_score(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))

plt.figure(figsize=(8,6))

plt.scatter(range(len(Y_test)), Y_test, color='green', label='Actual')

plt.scatter(range(len(Y_pred)), Y_pred, color='blue', label='Predicted')

plt.xlabel("Data Points")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices")
plt.legend()

plt.show()
