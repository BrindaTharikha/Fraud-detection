## Detecting Financial Fraud with Machine Learning: From EDA to EC2

### Overview

Financial fraud is a persistent threat that costs institutions billions each year. This article outlines a robust end to end ML pipeline I developed to detect fraudulent transactions from exploratory data analysis and preprocessing to model training, tuning, and finally deploying the solution using Docker on AWS EC2.

### Step 1: Exploratory Data Analysis (EDA)

The goal of EDA is to understand the dataset’s structure, imbalance, and potential predictive features.

Key Steps:

* Target Distribution: The dataset was severely imbalanced with less than 3 percent fraudulent samples. This was visualized using countplot
* Feature Distribution: Histograms for numeric columns like Transaction\_Amount, Account\_Balance, and Card\_Age highlighted skewness and required log transformations
* Outlier Detection: Boxplots revealed the presence of extreme outliers, especially in transaction amounts
* Correlation Analysis: Feature correlations with the fraud label showed that transaction patterns, time, and card behavior influenced the likelihood of fraud

EDA artifacts (saved as PNGs) helped validate assumptions and guided feature engineering

### Step 2: Preprocessing and Feature Engineering

Feature Engineering:

* Created Transaction\_Amount\_Log using a natural log transformation to reduce skewness in transaction values

Categorical Encoding:

* Applied one hot encoding to the following columns:

  * Transaction\_Type
  * Device\_Type
  * Location
  * Merchant\_Category
  * Card\_Type
  * Authentication\_Method

Scaling:

* StandardScaler was applied to these numerical features to normalize their ranges:

  * Transaction\_Amount\_Log
  * Account\_Balance
  * Avg\_Transaction\_Amount\_7d
  * Transaction\_Distance
  * Card\_Age

Final Dataset:

* Target column: Fraud\_Label
* Data was split into training and test sets using stratification to preserve class ratios
* Feature column names were saved to ensure compatibility during inference

### Step 3: Model Training and Evaluation

Logistic Regression (Baseline):

* Provided a simple, interpretable baseline for fraud classification
* Helped understand how a linear model performs on this imbalanced task

XGBoost Classifier (Main Model):

* Designed to capture complex nonlinear relationships
* Capable of handling class imbalance using scale\_pos\_weight
* Performed significantly better in precision, recall, and F1 score

Isolation Forest (Exploratory):

* Used for anomaly detection without requiring labels
* Did not perform well compared to supervised methods

### Step 4: Model Tuning and Cross Validation

Cross Validation:

* Used StratifiedKFold with 5 splits to preserve the imbalance ratio across folds
* RandomizedSearchCV was employed for efficient hyperparameter tuning

Tuned Hyperparameters for XGBoost:

* n\_estimators
* learning\_rate
* max\_depth
* subsample
* colsample\_bytree

Early Stopping:

* An internal validation set was carved out from the training data
* After each boosting round, validation performance was monitored
* Training stopped when performance stopped improving, which helped avoid overfitting and reduced unnecessary computation

### Step 5: Model Performance

| Metric            | Logistic Regression | XGBoost |
| ----------------- | ------------------- | ------- |
| Accuracy          | 0.92                | 0.96    |
| Precision (Fraud) | 0.34                | 0.71    |
| Recall (Fraud)    | 0.45                | 0.86    |
| F1 Score (Fraud)  | 0.38                | 0.78    |
| ROC AUC           | 0.73                | 0.94    |

XGBoost offered a better balance of precision and recall, which is vital in minimizing false negatives in fraud detection

### Step 6: Deployment with FastAPI, Docker and EC2

FastAPI:

* Developed an API endpoint for fraud prediction
* Designed a simple web interface with form input
* Used Pydantic for input validation and structured responses

Docker:

* Created a Dockerfile with all necessary dependencies and model files
* Ensured consistent local and cloud environments

```
docker build -t fraud-app .
docker run -d -p 8000:8000 fraud-app
```

AWS EC2:

* Provisioned an Ubuntu instance and installed Docker
* Opened port 8000 in the security group for access
* Cloned the repository and launched the container

Deployed App URL: (http://44.203.93.242:8000/)

### Conclusion

From rigorous EDA and smart feature engineering to robust model tuning and real time deployment, this project showcases how to build a production ready fraud detection pipeline using open source tools and cloud infrastructure

Stay tuned for more deep dives. If you liked this post, feel free to connect or ask questions

MachineLearning FraudDetection XGBoost FastAPI Docker AWS EC2 MLOps
