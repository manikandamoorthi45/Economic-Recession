# Economic-Recession
![Uploading image.png…]()


# 🏦 Economic Recession Prediction Project

### 📘 Overview
This repository contains a data science project focused on **predicting economic recessions** using key **macroeconomic indicators**.  
The project aims to analyze historical trends in the U.S. economy and forecast recession periods using machine learning techniques.  
The entire workflow — from data collection, feature engineering, model training, and evaluation — has been implemented in Python via Jupyter Notebook.

---

### 📊 Dataset Description
All datasets were **manually collected** from reliable sources such as the U.S. Federal Reserve (FRED) and economic databases.  
The indicators include both real and financial sector metrics, covering consumer behavior, production, and financial markets.

**Variables Used:**
- Consumer Price Index (CPI)
- Retail Sales
- Consumer Loans
- Wages
- Real GDP
- Employment Rate
- Industrial Production
- House Price Index
- Trade Balance
- S&P 500 Stock Price
- FED Interest Rate
- M2 Money Supply
- USREC (Target: 1 = Recession, 0 = No Recession)

---

### ⚙️ Data Preprocessing
1. **Data Cleaning:** Renamed columns, standardized date formats, and merged all indicators on the `DATE` column.  
2. **Handling Missing Values:** Forward-fill and interpolation techniques were used where appropriate.  
3. **Feature Engineering:** Combined multiple economic indicators into a single feature dataset (`df_final`).  
4. **Balancing the Dataset:** Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the recession vs non-recession samples.  
5. **Data Splitting:** Divided the dataset into training (75%) and testing (25%) sets using `train_test_split()`.

---

### 🤖 Models Used
Several classification algorithms were implemented and compared to predict whether the economy is entering a recession:

| Model | Description | Purpose |
|:------|:-------------|:--------|
| **Logistic Regression** | A baseline model for binary classification problems. | Interpretable model showing relationships between indicators and recession likelihood. |
| **Decision Tree Classifier** | Non-linear model capturing complex interactions. | Helps visualize key drivers of recessions. |
| **K-Nearest Neighbors (KNN)** | Distance-based classifier. | Provides alternative non-parametric view of relationships. |
| **Naïve Bayes** | Probabilistic model. | Used for comparison to simple statistical baseline. |

Model tuning was performed using **GridSearchCV** and **Cross-Validation** to optimize hyperparameters such as tree depth and leaf size.

---

### 📈 Evaluation Metrics
The following metrics were used to assess model performance:

- **Accuracy Score**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC Curve & AUC (Area Under Curve)**

Visualizations include:
- ROC Curve plots for each model  
- Confusion Matrix heatmaps using `seaborn`  
- Feature importance plots for tree-based models

---

### 📊 Key Findings & Insights
- Logistic Regression and Decision Tree models provided **consistent accuracy and interpretability**.  
- The **ROC AUC** values indicate good discrimination between recession and non-recession periods.  
- **SMOTE balancing** significantly improved recall for the minority (recession) class.  
- Economic indicators like **Real GDP, Employment Rate, and Industrial Production** were the **strongest predictors** of recession.  
- Cross-validation confirmed the model's robustness across different time periods.



---

### 🔍 How to Run the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/manikandamoorthi45/Economic-Recession.git
   ```
2. Open Jupyter Notebook and run:
   ```bash
   jupyter notebook "Final_project Economy Recession.ipynb"
   ```
3. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```

---

### 💡 Future Improvements
- Add time-series models (ARIMA, VAR, or LSTM) to capture sequential dependencies.  
- Integrate real-time API updates for macroeconomic indicators.  
- Expand the model to predict **recession severity or duration**.  
- Deploy the model via a web dashboard for live forecasts.

---

### 📜 Conclusion
This project demonstrates how macroeconomic data and machine learning can be combined to forecast economic recessions.  
By analyzing patterns in growth, inflation, employment, and financial indicators, the model provides early warning signals useful for policymakers and investors.

---

### 👨‍💻 Packages used 
Internship Project – Economic Recession Analysis  
📍 Developed using Python, scikit-learn, Jupyter Notebook

---

