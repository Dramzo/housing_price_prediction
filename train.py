import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df


def build_pipeline(numerical_features):

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ]
    )

    model = RandomForestRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def train_model(X_train, y_train, pipeline):

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:.3f}")
    print(f"R2 Score: {r2:.3f}")


def main():

    df = load_data()

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    numerical_features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(numerical_features)

    model = train_model(X_train, y_train, pipeline)

    print("Best Parameters:", model.best_params_)

    evaluate(model.best_estimator_, X_test, y_test)

    joblib.dump(model.best_estimator_, "random_forest_model.pkl")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()