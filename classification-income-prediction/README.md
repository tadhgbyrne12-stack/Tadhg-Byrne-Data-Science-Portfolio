# Classification Analysis on Real-World Demographic Data

## 1. Problem Overview
The goal of this project was to predict whether an individual earns more than $50,000 per year based on demographic and employment-related features. The task involved working with mixed categorical and numerical data and evaluating suitable classification approaches.

## 2. Data Overview
- **Dataset:** Adult Census Income dataset  
- **Target variable:** Income category (>50K / ≤50K)  
- **Features:** Age, education, marital status, occupation, hours worked per week, capital gains/losses  
- The dataset contains both numerical and categorical variables and reflects real-world socioeconomic bias

## 3. Data Preparation
- Handled missing values and inconsistent entries  
- Encoded categorical variables using appropriate encoding techniques  
- Scaled numerical features where required  
- Split data into training and testing sets to evaluate generalisation performance  

## 4. Modelling Approach
Two classification models were implemented and compared:

### Decision Tree Classifier
- Chosen for interpretability and ability to handle mixed data types  
- Key hyperparameters tuned included maximum tree depth  

### K-Nearest Neighbours (KNN)
- Chosen as a distance-based baseline classifier  
- Performance evaluated across multiple values of *k*  

## 5. Evaluation
Models were evaluated using accuracy, confusion matrices, and F1 score.

- The Decision Tree achieved approximately **83% accuracy**  
- KNN achieved approximately **76–77% accuracy**, depending on the value of *k*  
- The Decision Tree demonstrated stronger overall performance and interpretability  

## 6. Results & Interpretation
- Decision Trees handled categorical-heavy data more effectively than KNN  
- Features such as marital status, education level, and age had the strongest influence on predictions  
- KNN performance was sensitive to feature scaling and noise in the dataset  
- These results suggest that tree-based models are more suitable than distance-based approaches for income classification tasks involving mixed demographic data  

## 7. Limitations
- The dataset contains inherent socioeconomic bias, which may affect fairness  
- Income is influenced by external factors not captured in the dataset  
- Model performance may not generalise beyond the dataset context  

## 8. Next Steps
- Apply cross-validation for more robust evaluation  
- Explore ensemble methods such as Random Forests  
- Investigate fairness-aware modelling techniques
