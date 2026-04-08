House Price Prediction
A machine learning project that predicts house prices using Linear Regression based on features such as size, number of bedrooms, and location.

Overview
This project demonstrates a complete machine learning workflow:
Data preprocessing
Feature encoding
Model training
Performance evaluation
Visualization of results

The goal is to estimate house prices accurately using basic structured data.

Dataset
The dataset contains the following features:
size → Area of the house
bedrooms → Number of bedrooms
city → Location of the house
price → Target variable (house price)

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib

Model
Linear Regression
Suitable for predicting continuous values like house prices
Simple, fast, and interpretable

Workflow
Load dataset
Clean column names
Apply One-Hot Encoding for categorical data
Split dataset into training and testing sets
Train Linear Regression model
Make predictions
Evaluate using metrics
Visualize results

Evaluation Metrics
R² Score → Measures model accuracy
Mean Squared Error (MSE) → Measures prediction error

Results
The model produces predictions reasonably close to actual values
Performance improves with proper encoding and preprocessing

Visualization
Scatter plot comparing:
Actual prices (green points)
Predicted prices (blue points)


Future Improvements
Use advanced models (Random Forest, Gradient Boosting)
Add more features (location type, amenities, age of property)
Improve dataset size and quality

Conclusion
This project demonstrates how machine learning can be applied to real-world problems like house price prediction using simple regression techniques.

Author
Sanaya Muchhal
