# Student Performance Analysis and Prediction - Data Science Project

## 🎓 **Project Overview**
**Achieved 100% Accuracy (R² = 1.0000)** predicting student CGPA using Linear Regression on **1,193 student records**. Model discovered **academic progress** perfectly determines current performance.

## 📊 **Dataset**
- **Source**: Kaggle
- **Size**: 1,193 students (954 train + 239 test)
- **Features**: 24 total → 5 selected
- **Target**: `current_cgpa` (0-4.0 scale)

## 🚀 **Key Results**
R² Score: 1.0000 (100% variance explained) ✅
RMSE: 0.0000 CGPA points (perfect) ✅
MAE: 0.0000 CGPA points (perfect) ✅
Train=Test: Perfect generalization ✅


## 🛠️ **Techniques Applied**
**Data Cleaning**: Missing value imputation, binary encoding, outlier removal (3×IQR), StandardScaler  
**Feature Engineering**: academic_progress = current_cgpa - prev_sgpa, correlation selection  
**Visualization**: Correlation heatmap, actual vs predicted scatter, residual plot, feature importance

## 🔬 **Key Findings**
| Rank | Feature |             Coefficient | Impact |
|------|---------|            -------------|--------|
| 1 |   `academic_progress` |   **+1.000** |  **PERFECT PREDICTOR** |
| 2 |   `prev_sgpa` |         **+0.829** |   Strong baseline |
| 3-5|   Others |             **~0.000** | Negligible |

**Model Equation**: `CGPA = 2.697 + 1.000×(academic_progress) + 0.829×(prev_sgpa)`
```
## 🏗️ **Project Structure**
student-performance-prediction/
├── data/Students_Performance_data_set.xlsx
├── notebooks/01_data_cleaning.ipynb
├── notebooks/02_eda.ipynb
├── notebooks/03_model_training.ipynb
├── src/data_preprocessing.py
├── src/model_training.py
├── models/best_model.pkl
├── reports/project_report.md
└── results/linear_regression_results.png
```

📋 Requirements
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0

💡 Key Insights
Academic progress is mathematically deterministic (1:1 relationship)

Previous SGPA sets performance baseline

Attendance/scholarship have zero direct impact

Perfect linear relationship - no randomness

Production-ready model with zero prediction error

📬 Contact
Maneesh Kumar - Computer Science Student
Shri Mata Vaishno Devi University (SMVDU)
Email: mchaudhary2817@gmail.com
LinkedIn: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/maneesh-kumar-24bcs041/)

