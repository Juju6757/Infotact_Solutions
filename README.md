# NYC Airbnb Price Prediction & Geospatial Analysis

## Overview

This project explores the **AB_NYC_2019.csv** dataset to uncover insights about Airbnb listings in New York City. The workflow covers everything from exploratory data analysis (EDA) and data cleaning to geospatial visualization and baseline machine learning for price prediction.

The primary goal is to predict Airbnb listing prices using property attributes and engineered spatial features, specifically leveraging an XGBoost regression model.

## Project Structure & Files

* **AB_NYC_2019.csv**: The core dataset containing 2019 NYC Airbnb listings, including host details, neighborhood information, pricing, availability, and geographical coordinates.
* **New_York_City_.png**: A generated geographical plot/heatmap visualizing the distribution of the Airbnb listings across the different NYC boroughs.
* **Notebook Code**: The Python environment/notebook containing the data processing, feature engineering, and model training steps.

## Workflow

### 1. Data Cleaning & Preprocessing

* **Handling Missing Values**: Rows with missing critical information (like `name`, `host_name`, and `neighbourhood_group`) are dropped. Missing `reviews_per_month` are filled with `0`, and missing `last_review` dates are filled with a placeholder string.
* **Outlier Removal**: The price distribution is highly skewed by extreme values. The Interquartile Range (IQR) method is applied to filter out pricing outliers, ensuring a more normalized dataset for the machine learning model.

### 2. Geospatial Analysis

* Using `geopandas` and `shapely`, the spatial coordinates (`latitude` and `longitude`) of the listings are converted into point geometries.
* This allows for mapping the density and geographical spread of listings across New York City (as seen in **New_York_City_.png**).

### 3. Feature Engineering

* **Distance Calculation**: A custom `haversine` function is implemented to calculate the distance (in kilometers) from each listing to the geographical center of NYC (coordinates of the Empire State Building).
* This new feature (`distance_to_center_km`) provides the machine learning model with valuable context regarding a property's proximity to downtown Manhattan, which strongly influences price.

### 4. Machine Learning Pipeline

* **Features Used**: `neighbourhood_group`, `room_type`, `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count`, `availability_365`, and `distance_to_center_km`.
* **Preprocessing**: `scikit-learn`'s `ColumnTransformer` is used to apply One-Hot Encoding to categorical variables while passing numeric features through directly. Missing values in test/train splits are handled using median and mode imputations.
* **Model**: A baseline `XGBRegressor` (XGBoost) model is trained on an 80/20 train-test split to predict the target variable (`price`).

## Baseline Results

* **RMSE (Root Mean Squared Error)**: ~217.89
* **MAPE (Mean Absolute Percentage Error)**: ~94,311,366,524,928.00 *(Note: This highly inflated MAPE is indicative of near-zero actual prices in the dataset causing division-by-zero-like spikes, which will require further target-variable filtering in future iterations).*

## Dependencies

To run this project, you will need the following Python libraries installed:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `haversine`
* `geopandas`
* `shapely`
* `scikit-learn`
* `xgboost`
