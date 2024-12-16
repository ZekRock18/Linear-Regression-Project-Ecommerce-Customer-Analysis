# Linear Regression Project

## Overview
This project analyzes customer data for an Ecommerce company to help determine whether the company should focus on improving its mobile app experience or website. The analysis involves data exploration, visualization, and building a linear regression model to predict customer spending.

---

## Dataset
The dataset includes the following columns:
- **Avg. Session Length**: Average time spent in a session.
- **Time on App**: Time spent on the mobile app.
- **Time on Website**: Time spent on the website.
- **Length of Membership**: Duration of customer membership.
- **Yearly Amount Spent**: Total yearly spending by the customer.

---

## Steps Performed

### 1. Imported Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```

### 2. Loaded the Dataset
```python
customers = pd.read_csv("Ecommerce Customers.csv")
```

### 3. Data Exploration
- Displayed basic information and summary statistics.
- Visualized relationships between features using pair plots and joint plots.

#### Key Insights:
- **Length of Membership** had the strongest correlation with yearly spending.

### 4. Linear Regression Analysis
#### a. Feature Selection
```python
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
```
#### b. Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```
#### c. Model Training
```python
lm = LinearRegression()
lm.fit(X_train, y_train)
```
#### d. Model Coefficients
| Feature                | Coefficient |
|------------------------|-------------|
| Avg. Session Length    | 25.981550   |
| Time on App            | 38.590159   |
| Time on Website        | 0.190405    |
| Length of Membership   | 61.279097   |

#### Interpretation:
- **Time on App**: A 1-unit increase results in a $38.59 increase in yearly spending.
- **Time on Website**: Minimal impact ($0.19 increase per unit).
- **Length of Membership**: The most significant feature ($61.28 increase per unit).

### 5. Predictions
Predicted values were compared with actual values using a scatter plot. 

### 6. Model Evaluation
- **Mean Absolute Error (MAE)**: 7.82
- **Mean Squared Error (MSE)**: 79.81
- **Root Mean Squared Error (RMSE)**: 8.93

### 7. Residual Analysis
Residuals were normally distributed, indicating a good model fit.

---

## Conclusion
- **Mobile App**: Significant impact on yearly spending (38.59 coefficient).
- **Website**: Minimal contribution (0.19 coefficient).
- **Length of Membership**: Most influential feature (61.28 coefficient).

### Recommendation
- Focus on enhancing the **mobile app experience**.
- Increase efforts to retain long-term customers to maximize spending.

---

## How to Run
1. Clone the repository.
2. Ensure the required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place the `Ecommerce Customers.csv` file in the same directory as the script.
4. Run the Python script to view the results and visualizations.
