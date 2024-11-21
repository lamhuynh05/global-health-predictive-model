import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from math import sqrt

# Step 1: Load the dataset
file_path = 'SDG_goal3_clean.csv'
data = pd.read_csv(file_path)

# One-hot encode the 'Region' categorical variable
data = pd.get_dummies(data, columns=['Region'], drop_first=True)

# Select the features and target variable (UHC Index)
X = data[['Maternal mortality ratio',
          'Proportion of births attended by skilled health personnel (%)',
          'Infant mortality rate (deaths per 1,000 live births):::BOTHSEX',
          'Under-five mortality rate, by sex (deaths per 1,000 live births):::BOTHSEX',
          'Health worker density, by type of occupation (per 10,000 population)::PHYSICIAN'] + 
          list(data.filter(regex='Region_').columns)]

# Target variable: UHC Index
y = data['Universal health coverage (UHC) service coverage index']

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (for better performance in linear regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr_model.predict(X_test)

# Evaluate the model using MSE, RMSE, MAE, and R-squared
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression - Mean Squared Error (MSE): {mse:.2f}')
print(f'Linear Regression - Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Linear Regression - Mean Absolute Error (MAE): {mae:.2f}')
print(f'Linear Regression - R-squared: {r2:.2f}')

# Display the coefficients for each feature (for model interpretation)
coefficients = pd.DataFrame(lr_model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Visualize the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.xlabel('Actual UHC Index')
plt.ylabel('Predicted UHC Index')
plt.title('Linear Regression - Actual vs Predicted UHC Index')
plt.show()



# Linear Regression - Mean Squared Error (MSE): 69.67
# Linear Regression - Root Mean Squared Error (RMSE): 8.35
# Linear Regression - Mean Absolute Error (MAE): 7.06
# Linear Regression - R-squared: 0.46

#                                                     Coefficient
# Maternal mortality ratio                               2.011373
# Proportion of births attended by skilled health...     0.596125
# Infant mortality rate (deaths per 1,000 live bi...   -20.973701
# Under-five mortality rate, by sex (deaths per 1...    11.632594
# Health worker density, by type of occupation (p...     3.271366
# Region_Americas                                        3.282539
# Region_Asia                                            4.096150
# Region_Europe                                          2.669380
# Region_Oceania                                         3.605392

