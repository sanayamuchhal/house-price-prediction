# House Price Prediction using Linear Regression

A machine learning project that predicts house prices using Linear Regression based on features such as property size, number of bedrooms, and location.



# Overview

This project demonstrates a complete end-to-end machine learning workflow, including:
* Data preprocessing and cleaning
* One-Hot Encoding
* Model training using Linear Regression
* Model evaluation using standard metrics
* Visualization of prediction performance

The goal is to build a simple yet effective model for estimating house prices using structured data.

# Dataset
The dataset consists of the following features:
* size → Area of the house
* bedrooms → Number of bedrooms
* city → Location of the house
* price→ Target variable (house price)

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib


# Model

*Linear Regression

* Suitable for predicting continuous values
* Simple, fast, and interpretable
* Serves as a strong baseline model

# Workflow

1. Load dataset
2. Clean and preprocess data
3. Perform feature engineering (`size_per_bedroom`)
4. Reduce categorical complexity (top cities grouping)
5. Apply One-Hot Encoding
6. Split dataset into training and testing sets
7. Train Linear Regression model
8. Make predictions on test data
9. Evaluate model performance
10. Visualize actual vs predicted values


# Evaluation Metrics

* R² Score → Measures how well the model explains variance
* Mean Squared Error (MSE) → Measures average prediction error

  
# Visualization

The model performance is visualized using a custom comparison plot:

* green- Actual Prices
* blue- Predicted Prices
* Vertical lines representing prediction error

This visualization clearly shows how close predictions are to actual values for each house



# Results

* The model produces predictions reasonably close to actual values
* Feature engineering improves prediction quality
* Visualization confirms alignment between actual and predicted values


# Future Improvements

* Implement advanced models 
* Improve accuracy
* Better visualization
* Increase dataset size for better generalization


# Conclusion

This project demonstrates how machine learning can be applied to real-world problems like house price prediction using a simple and interpretable regression model. It highlights the importance of preprocessing, feature engineering, and visualization in building effective models.


# Author
Sanaya Muchhal


# Project Status

*This project is actively being improved and updated.*


