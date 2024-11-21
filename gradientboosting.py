import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import sqrt

# Step 1: Load the dataset
file_path = 'SDG_goal3_clean.csv'
data = pd.read_csv(file_path)

# One-hot encode the 'Region' and 'Country' categorical variables
data = pd.get_dummies(data, columns=['Region', 'Country'], drop_first=True)

# Select the features and target variable (UHC Index)
X = data[['Maternal mortality ratio',
          'Proportion of births attended by skilled health personnel (%)',
          'Infant mortality rate (deaths per 1,000 live births):::BOTHSEX',
          'Under-five mortality rate, by sex (deaths per 1,000 live births):::BOTHSEX',
          'Health worker density, by type of occupation (per 10,000 population)::PHYSICIAN',
          'Health worker density, by type of occupation (per 10,000 population)::NURSEMIDWIFE',
          'Adolescent birth rate (per 1,000 women aged 15-19 years)',
          'Mortality rate attributed to cardiovascular disease, cancer, diabetes or chronic respiratory disease (probability):::BOTHSEX'] + list(data.filter(regex='Region_').columns)]

# Target variable: UHC Index
y = data['Universal health coverage (UHC) service coverage index']

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Gradient Boosting model with hyperparameter tuning
gb_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model using MSE, RMSE, MAE, and R-squared
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = sqrt(mse_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'Gradient Boosting - Mean Squared Error (MSE): {mse_gb:.2f}')
print(f'Gradient Boosting - Root Mean Squared Error (RMSE): {rmse_gb:.2f}')
print(f'Gradient Boosting - Mean Absolute Error (MAE): {mae_gb:.2f}')
print(f'Gradient Boosting - R-squared: {r2_gb:.2f}')

# Feature importance
feature_importances_gb = pd.DataFrame(gb_model.feature_importances_, index=X.columns, 
                        columns=['Importance']).sort_values('Importance', ascending=False)
print("Feature Importances (Gradient Boosting):\n", feature_importances_gb)

# Visualize actual vs predicted values
plt.scatter(y_test, y_pred_gb)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.xlabel('Actual UHC Index')
plt.ylabel('Predicted UHC Index')
plt.title('Gradient Boosting - Actual vs Predicted UHC Index')
plt.show()

# Gradient Boosting - Mean Squared Error (MSE): 37.78
# Gradient Boosting - Root Mean Squared Error (RMSE): 6.15
# Gradient Boosting - Mean Absolute Error (MAE): 4.80
# Gradient Boosting - R-squared: 0.70
# Feature Importances (Gradient Boosting):
#                                                      Importance
# Infant mortality rate (deaths per 1,000 live bi...    0.326469
# Under-five mortality rate, by sex (deaths per 1...    0.223937
# Mortality rate attributed to cardiovascular dis...    0.175894
# Health worker density, by type of occupation (p...    0.081439
# Maternal mortality ratio                              0.080673
# Health worker density, by type of occupation (p...    0.056040
# Proportion of births attended by skilled health...    0.021091
# Adolescent birth rate (per 1,000 women aged 15-...    0.012940
# Region_Europe                                         0.010146
# Region_Americas                                       0.008729
# Region_Asia                                           0.001731
# Region_Oceania                                        0.000912
