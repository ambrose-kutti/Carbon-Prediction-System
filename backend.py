import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

df = pd.read_csv("CO2 Emissions_Canada.csv")    # Load Dataset

# Feature selection
features = ['Engine Size(L)', 'Cylinders', 'Fuel Type', 'Fuel Consumption Comb (mpg)']
target = 'CO2 Emissions(g/km)'

X = df[features]
y = df[target]

# Preprocessing
numeric_features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (mpg)']
categorical_features = ['Fuel Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

os.makedirs("model", exist_ok=True)    # Create model directory if not exists

# 1. Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X, y)
joblib.dump(rf_pipeline, 'model/rf_model.pkl')

# 2. Support Vector Regression
svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf', C=100, epsilon=1.0))
])
svr_pipeline.fit(X, y)
joblib.dump(svr_pipeline, 'model/svr_model.pkl')

# 3. ANN Model
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

ann_model.compile(optimizer='adam', loss='mean_absolute_error')
ann_model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[EarlyStopping(patience=10)], verbose=0)
ann_model.save('model/ann_model.h5')

# Evaluation (optional)
rf_pred = rf_pipeline.predict(X)
print("Random Forest R2 Score:", r2_score(y, rf_pred))

svr_pred = svr_pipeline.predict(X)
print("SVR R2 Score:", r2_score(y, svr_pred))

ann_pred = ann_model.predict(X_processed)
print("ANN R2 Score:", r2_score(y, ann_pred))
