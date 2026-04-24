
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Step 2: Remove invalid data
df = df[(df['price'] > 0) & (df['size'] > 0) & (df['bedrooms'] > 0)]
df = df.dropna()

df['size_per_bedroom'] = df['size'] / df['bedrooms']


# Step 3: Reduce city categories
top_cities = df['city'].value_counts().nlargest(3).index
df['city'] = df['city'].apply(lambda x: x if x in top_cities else 'Other')

# Step 4: One-Hot Encoding
X = pd.get_dummies(df[['size', 'bedrooms', 'size_per_bedroom', 'city']], drop_first=True)


Y = df['price']

# Step 5: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
     X, Y, test_size=0.2, random_state=42
)

# Step 6: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

print("Model trained")
Y_pred = model.predict(X_test)
Y_test_actual = Y_test

# Step 10: Evaluation
print("\nR2 Score:", r2_score(Y_test_actual, Y_pred))
print("MSE:", mean_squared_error(Y_test_actual, Y_pred))

# Step 7: Comparison output
compare = pd.DataFrame({
    'Actual': Y_test_actual.round().astype(int),
    'Predicted': Y_pred.round().astype(int)
})
print("\nActual vs Predicted:\n", compare.head())


n = 30  # number of data points to show

plt.figure(figsize=(10,6))

# vertical lines (error)
for i in range(n):
    plt.plot([i, i], [Y_test_actual.iloc[i], Y_pred[i]],
             color='gray', alpha=0.5)

# actual points (green)
plt.scatter(range(n), Y_test_actual.iloc[:n],
            color='green', label='Actual')

# predicted points (blue)
plt.scatter(range(n), Y_pred[:n],
            color='blue', label='Predicted')

plt.xlabel("Data Points (Each = One House)")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices")
plt.legend()

plt.show() 
