# 📈 Linear & Polynomial Regression – Advertising Sales Prediction

A machine learning project that predicts product **sales** based on advertising spend across TV, Radio, and Newspaper channels. The project covers exploratory data analysis, simple multiple linear regression, and polynomial regression with degree optimization.

---

## 📁 Dataset

**`Advertising.csv`** — Contains advertising budgets and resulting sales figures.

| Column | Description |
|--------|-------------|
| `TV` | Budget spent on TV ads |
| `radio` | Budget spent on Radio ads |
| `newspaper` | Budget spent on Newspaper ads |
| `sales` | Product sales (target variable) |

---

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Loaded dataset using **Pandas**
- Plotted individual scatter plots for each feature vs. sales using **Matplotlib**
- Generated a **pairplot** using **Seaborn** to visualize feature relationships

### 2. Statistical Summary (OLS)
- Used **Statsmodels** (`smf.ols`) to fit an OLS regression model
- Formula: `sales ~ TV + radio + newspaper`
- Reviewed model summary for coefficients, R², p-values

### 3. Multiple Linear Regression (Scikit-learn)
- Split data: **70% train / 30% test** (`random_state=42`)
- Trained a `LinearRegression` model
- Evaluated using:
  - **MAE** (Mean Absolute Error)
  - **MSE** (Mean Squared Error)
  - **RMSE** (Root Mean Squared Error)

### 4. Polynomial Regression
- Applied `PolynomialFeatures` (degree=2) to create interaction and squared terms
- Compared train vs. test RMSE across polynomial degrees 1–9 to detect overfitting
- Selected **degree=3** as the optimal complexity based on the RMSE curve

### 5. Model Persistence
- Saved the final model and polynomial converter using **Joblib**
- Reloaded and used the model to predict sales for a new campaign

---

## 📊 Results

| Model | MAE | RMSE |
|-------|-----|------|
| Multiple Linear Regression | ~1.46 | ~1.74 |
| Polynomial Regression (degree=2) | ~0.48 | ~0.63 |
| **Final Model (degree=3)** | ✅ Best fit | ✅ Lowest error |

> Mean sales value in dataset: ~14.02

---

## 🚀 Sample Prediction

```python
campaign = [[439, 22, 12]]  # TV=439, Radio=22, Newspaper=12
campaign_poly = loaded_poly.transform(campaign)
predicted_sales = final_model.predict(campaign_poly)
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data loading & manipulation |
| `matplotlib` | Data visualization |
| `seaborn` | Pairplot & statistical plots |
| `statsmodels` | OLS regression summary |
| `scikit-learn` | ML models, metrics, train-test split |
| `joblib` | Model serialization |

---

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/linear-regression-advertising.git
   cd linear-regression-advertising
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn statsmodels scikit-learn joblib
   ```

3. **Add the dataset**  
   Place `Advertising.csv` in the project root directory.

4. **Run the script**
   ```bash
   python linera_regression.py
   ```

---

## 📌 Key Takeaways

- **TV** advertising has the strongest positive correlation with sales
- **Newspaper** spend shows the weakest relationship
- Polynomial features (degree=3) significantly improve prediction accuracy over simple linear regression
- Overfitting becomes evident beyond degree=4 when test RMSE starts rising

---

## ⚠️ Known Issues / Notes

- There is a minor **indentation bug** in the plotting loop (lines with `plt.plot` inside the for-loop are incorrectly indented — this should be outside the loop)
- The file is named `linera_regression.py` (typo — should be `linear_regression.py`)

---


