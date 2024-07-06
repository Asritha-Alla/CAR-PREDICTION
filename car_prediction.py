import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('car data.csv')
print(data.info())
print("\nSample data:")
print(data.head())

data = data.drop(['Car_Name'], axis=1)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_preprocessed, y_train)

y_pred = rf_model.predict(X_test_preprocessed)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

importance = rf_model.feature_importances_
feature_names = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


## use to see the feature importance plot
# plt.figure(figsize=(10, 6))
# plt.bar(feature_importance['feature'], feature_importance['importance'])
# plt.xticks(rotation=90)
# plt.title('Feature Importance')
# plt.tight_layout()
# plt.show()
# plt.close('all')


## use to see the Actual vs Predicted Car Prices plot
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Actual vs Predicted Car Prices')
# plt.tight_layout()
# plt.show()
# plt.close('all')

def predict_price(year, present_price, driven_kms, fuel_type, seller_type, transmission, owner):
    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Driven_kms': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Selling_type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })

    input_preprocessed = preprocessor.transform(input_data)

    predicted_price = rf_model.predict(input_preprocessed)

    return predicted_price[0]

predicted_price = predict_price(2015, 5.59, 27000, 'Petrol', 'Dealer', 'Manual', 0)
print(f"\nPredicted price for the new car: {predicted_price:.2f}")