# Global Health Metrics Predictive Models

## Linear Regression Model:
Linear Regression was selected as the baseline model due to its simplicity and interpretability. The linear regression method assumes a linear relationship between the input attributes and the predicted variable which is UHC index in this analysis. Despite the complexity of the healthcare industry, the Linear Regression model is incredibly useful as it provides a benchmark and helps us understand basic linear correlation between the selected healthcare indicators and the UHC index. 

<img width="656" alt="Screenshot 2024-11-22 at 1 57 43 am" src="https://github.com/user-attachments/assets/681ee8ad-6828-493c-be13-eb86860ba7d3">

- R-squared (R²): This linear regression model has an R² of 0.46 which means that the model can only explain 46% of the variance of the UHC index, suggesting that the model is underfitting the data. Underfitting happens when a model is too simple to analyse the complexity of the relationship between the features and target variable. In this case, since healthcare system and related outcomes are influenced by many different factors which linear regression can not account for. 

- RMSE and MAE: The RMSE of 8.35 demonstrates how this model’s predicted values deviate from the actual UHC index values by about 8 points on average. On the other hand, the MAE of 7.06 illustrates how the predictions are off by approximately 7 points. The error metrics are moderately high which further shows that this model does not capture the complexity of the data. 

- Infant mortality rate (-20.97): Infant mortality rate has a negative coefficient which shows a negative relationship between it and the UHC index which aligns with how poor infant health affects healthcare quality in real life. 

- Health worker density (Physician) (3.27): Health worker density has a positive coefficient which demonstrates how more health workers would contribute more to the UHC index. 
Proportion of births attended by health personnel (0.59): Even though this coefficient is relatively low compared to other coefficients, it is still positive which indicates a positive relationship between the proportion and the UHC index. 

Overall, the Linear Regression model has a poor performance on predicting UHC index values. Therefore, we need a complex machine learning model to capture the possible non-linear correlations of the data. 

## Gradient Boosting Model:
Gradient Boosting was selected as the second method because of its ability to show non-linear correlations between features and target variables. Gradient Boosting builds an ensemble of decision trees where each tree corrects the errors from the previous trees, which can overall improve the model’s performance on complicated dataset such as this healthcare one.  

The same input attributes were used as the Linear Regression but these attributes are also included: 
- Health worker density (Nurse/Midwife)
- Adolescent birth rate
- Mortality rate due to cardiovascular diseases

<img width="645" alt="Screenshot 2024-11-22 at 1 59 33 am" src="https://github.com/user-attachments/assets/f53b09c2-17b9-476d-9ccf-f662b97c7952">

- R-squared (R²): An R-squared value of 0.70 means that the Gradient Boosting model can account for 70% of the variance in the UHC index which is significantly better than the performance of Linear Regression model. This means that Gradient Boosting can explain complex, non-linear correlation between the features and target variable, which makes it a more appropriate model for this analysis. 

- RMSE and MAE: The RMSE of 6.15 illustrates that the model's predicted values deviate from the actual UHC Index values by approximately 6 points on average, which is significantly lower than the RMSE for Linear Regression (8.35). The MAE of 4.80 shows that the typical prediction error is much smaller than the MAE of Linear Regression, which further indicates better prediction performance.

Gradient Boosting is a good predictive model to avoid overfitting, which is a common risk when using complex models with many parameters. By carefully tuning the hyperparameters (e.g., n_estimators = 150, learning_rate = 0.05, max_depth = 3), the model can learn the underlying patterns without overfitting to the training data. The lower MAE and RMSE, along with the high R², indicate that the model generalises well and is not overfitting the training data.

The Gradient Boosting model provides helpful insights of the importance of features when predicting UHC index:

- Infant mortality rate has a 32.6% importance, making it the most important feature. This aligns with real-world expectations, as countries with high infant mortality rates tend to have lower UHC Index scores.

- Under-five mortality rate has a 22.4% importance which highlights the critical role of pediatric healthcare in determining the UHC Index.

- Mortality rate due to cardiovascular diseases has a 17.6% importance and reflects the significance of diseases in shaping healthcare outcomes.

- Health worker density (Physician) has a 8.1% importance, which is relatively high and indicates that the density of physicians plays a crucial role in determining the quality of healthcare services.

Overall, the Gradient Boosting method is better than Linear Regression when it comes to capturing non-linear relationships with multiple attributes. 
