# DTSC5082_Group11_Delivery-Delay-Predictor
Delivery-time prediction using machine learning, feature engineering, and SHAP explainability. DTSC 5082 course project.

## Project Overview

This project investigates the following research question:

> Can machine learning predict delivery time and identify deliveries with relatively high delivery-time risk using order, agent, traffic, weather, and location information?

The workflow includes data-quality auditing, cleaning, exploratory analysis, feature engineering, regression modeling, cross-validation, hyperparameter tuning, and SHAP explainability.

The dataset does not contain promised delivery deadlines. Therefore, the project predicts delivery duration and defines relative duration-based risk groups; it does not classify confirmed late deliveries.

## Repository Structure

```text
DTSC5082_Group11_Delivery-Delay-Predictor/
├── README.md
├── Group11_Delivery_Delay_Predictor.ipynb
└── data/
    └── amazon_delivery.csv
```

The notebook and the `data` directory should be located at the same directory level.

## Dataset

**Source:** [Amazon Delivery Dataset by Sujal Suthar on Kaggle](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset/data)

- Original dataset: 43,739 observations and 16 variables
- Cleaned dataset: 43,648 observations
- Target: `Delivery_Time`, measured in minutes
- Inputs include agent attributes, coordinates, timestamps, weather, traffic, vehicle type, delivery area, and product category.

Download `amazon_delivery.csv` from the source and place it in the `data/` directory before running the notebook.

## Data Preparation and Feature Engineering

The notebook audits missing values, duplicate records, categorical labels, numerical ranges, timestamps, and geographic coordinates.

Key preparation steps include:

- Normalizing categorical labels and textual missing-value placeholders.
- Removing 91 records with multiple simultaneous inconsistencies.
- Correcting 156 negative store-latitude values based on observed coordinate patterns.
- Retaining and documenting 3,495 records with zero store coordinates.
- Handling remaining missing agent ratings within the training pipeline.

Five features are engineered:

| Feature | Description |
|---|---|
| `Delivery_Distance_km` | Haversine distance between store and destination |
| `Pickup_Wait_Minutes` | Difference between order and pickup times, with midnight adjustment |
| `Order_Hour` | Hour of order placement |
| `Order_DayOfWeek` | Day of the week |
| `Is_Weekend` | Weekend indicator |

Actual pickup time is required to calculate `Pickup_Wait_Minutes`. The evaluated model therefore requires information available after pickup.

## Installation

Create and activate a Python virtual environment from the repository root.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn xgboost shap jupyterlab
```

These commands install unpinned dependencies. Package versions can affect numerical results; record the versions from a successfully tested environment when preparing a reproducible release.

## Running the Analysis

1. Place the dataset at `data/amazon_delivery.csv`.
2. Start JupyterLab from the repository root:

   ```bash
   python -m jupyterlab
   ```

3. Open `Group11_Delivery_Delay_Predictor.ipynb`.
4. Select the Python environment containing the installed dependencies.
5. Restart the kernel and run all cells in order.

The complete run includes cross-validation, randomized hyperparameter searches, and SHAP analysis. Execution time depends on available hardware.
Tables and plots are displayed within the notebook.

## Modeling and Evaluation

The project compares:

- Mean-based Dummy Regressor
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Regression
- Multilayer Perceptron

An 80/20 random split produces 34,918 training observations and 8,730 test observations, using random state 42.
Preprocessing is fitted within model pipelines and includes numerical imputation and scaling, categorical imputation, and one-hot encoding.
XGBoost, Random Forest, and MLP undergo five-fold cross-validation on the training set. Random Forest and XGBoost are then tuned using randomized search with 25 configurations per model.

Evaluation metrics are:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

## Reported Results

The following values are recorded in the notebook's saved outputs.

| Model | MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| Dummy Baseline | 41.704 | 52.121 | ～0 |
| Tuned XGBoost | 17.092 | 22.041 | 0.821 |
| Tuned Random Forest | **16.967** | **21.929** | **0.823** |

Tuned Random Forest reduced MAE by approximately 59.32% relative to the mean-based baseline and was selected for risk grouping and SHAP analysis.
These results describe performance within the evaluated dataset. 

## Relative Delivery-Time Risk

Risk thresholds are derived from the final model's training predictions.

| Risk Group | Predicted Delivery Time |
|---|---|
| Low Risk | At or below 119.66 minutes |
| Medium Risk | Above 119.66 and at or below 161.00 minutes |
| High Risk | Above 161.00 minutes |

The notebook applies the underlying calculated thresholds; the values above are rounded for presentation.
These groups describe relative predicted duration, not confirmed lateness or probabilities of missing a delivery deadline.

## Explainability

SHAP analysis examines the final Random Forest using a reproducible sample of 1,000 test observations.

Outputs include:

- Global importance of transformed features
- Importance aggregated by original predictor
- A SHAP beeswarm plot
- A local explanation for an illustrative high-risk delivery

SHAP explains model behavior and should not be interpreted as evidence of causal relationships.
