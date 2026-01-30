# Regression Analysis of Body Weight from Skeletal Measurements

## 1. Problem Overview
The objective of this project was to model the relationship between skeletal body measurements and an individual’s body weight using linear regression techniques. The aim was to assess how well bone dimensions can predict weight and to identify which measurements contribute most strongly to the prediction.

---

## 2. Data Overview
- Dataset containing numeric skeletal measurements only  
- Predictor variables include biacromial breadth, pelvic breadth, bitrochanteric breadth, chest depth, chest diameter, elbow diameter, wrist diameter, knee diameter, and ankle diameter  
- Target variable: **Body weight (kg)**  

The dataset represents anthropometric measurements and is suitable for linear regression due to its continuous numerical features.

---

## 3. Modelling Approach
Two regression approaches were applied:

- **Linear Regression** using scikit-learn for baseline modelling and cross-validation
- **Ordinary Least Squares (OLS)** using statsmodels to enable coefficient interpretation, statistical significance testing, and diagnostic analysis

Cross-validation was used to assess generalisation performance and reduce sensitivity to individual train/test splits.

---

## 4. Evaluation Metrics
Model performance was evaluated using:
- R-squared and adjusted R-squared  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  

---

## 5. Results
- Average cross-validated **R-squared ≈ 0.88 ± 0.01**
- Average cross-validated **RMSE ≈ 4.69 ± 0.36 kg**
- Results indicate strong explanatory power and stable generalisation across folds

The OLS model showed improved interpretability compared to standard linear regression, while maintaining strong predictive performance.

---

## 6. Interpretation & Insights
- Chest depth, chest diameter, and knee diameter were the strongest predictors of body weight  
- These variables showed statistically significant coefficients with high t-statistics and low p-values  
- Some variables, such as biacromial breadth and ankle diameter, showed weak or non-significant relationships with weight  

The findings align with anatomical intuition, as larger and denser skeletal structures contribute more substantially to overall body mass.

---

## 7. Limitations
- The dataset size limits the strength of generalisation  
- The model assumes linear relationships between skeletal measurements and weight  
- Other non-skeletal factors influencing body weight are not included  

---

## 8. Next Steps
- Test regularised regression methods (Ridge, Lasso, Elastic Net)  
- Expand analysis to additional populations  
- Investigate non-linear modelling approaches  
