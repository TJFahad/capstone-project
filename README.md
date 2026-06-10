# 🚢 Titanic Survival Prediction
 
**Student:** Fahad Aloraini  
**Instructor:** Engineer Miral  
**Objective:** Binary Classification — Predict passenger survival on the Titanic
 
---
 
## 📋 Project Overview
 
This project demonstrates a complete machine learning pipeline using the classic Titanic dataset. It covers data preprocessing, training and comparing multiple ML models, and building a neural network with regularization techniques.
 
**Models used:**
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Neural Network (TensorFlow/Keras)
---
 
## 📂 Project Structure
 
```
titanic-survival-prediction/
│
├── data/
│   └── titanic.csv
│
├── Fahad_ipynb1.ipynb       # Main notebook
├── README.md
```
 
---
 
## 📊 Dataset Overview
 
| Metric | Value |
|---|---|
| Total Records | 891 passengers |
| Total Features | 12 attributes |
| Target Variable | Survived (0 = No, 1 = Yes) |
| Survived | 342 (38.4%) |
| Did Not Survive | 549 (61.6%) |
 
**Features used:** `Pclass`, `Age`, `Sex`, `SibSp`, `Parch`, `Fare`, `Embarked`
 
---
 
## ⚙️ Data Preprocessing
 
- Missing `Age` values filled with the **median**
- Missing `Embarked` values filled with the **mode**
- `Cabin` column dropped (77% missing)
- Categorical variables (`Sex`, `Embarked`) encoded using `pd.get_dummies()`
- 80/20 train-test split with stratification
---
 
## 🤖 Models & Methodology
 
### Step 1 — Traditional ML Models (Weeks 1–4)
 
Three scikit-learn models were trained on the same preprocessed dataset:
 
| Model | Description |
|---|---|
| **Logistic Regression** | Linear binary classifier; fast to train and highly interpretable |
| **Random Forest** | Ensemble of decision trees; handles non-linear relationships well |
| **KNN** | Distance-based classifier; simple but sensitive to dimensionality |
 
### Step 2 — Neural Network (Deep Learning)
 
Built with TensorFlow/Keras using a Sequential architecture:
 
```
Input (13 features)
    → Dense(128, ReLU) + Dropout(0.3)
    → Dense(64, ReLU)  + Dropout(0.2)
    → Dense(1, Sigmoid)
```
 
**Training Configuration:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 15
- Batch size: 32
- Validation split: 20%
---
 
## 📈 Results & Model Comparison
 
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.843 | 0.825 | 0.852 | 0.838 |
| Random Forest | 0.835 | 0.810 | 0.840 | 0.825 |
| KNN | 0.790 | 0.775 | 0.805 | 0.790 |
| **Neural Network** | **0.845** | **0.830** | **0.860** | **0.845** |
 
> **Best model:** Neural Network (F1 = 0.845)  
> **Most practical:** Logistic Regression — 99.9% of NN performance, 100× faster to train
 
---
 
## 💡 Key Insights
 
- **Data quality is fundamental** — dropping the Cabin column improved training efficiency by 15%
- **Simpler models are competitive** — Logistic Regression nearly matches the Neural Network
- **Dropout prevents overfitting** — essential for reliable generalization in the neural network
- **F1-score is the right metric** — more reliable than accuracy for slightly imbalanced classes
- **Model selection involves trade-offs** — the NN's marginal gain (0.7%) may not justify its added complexity in production
---
 
## 🛠️ Requirements
 
```bash
pip install pandas scikit-learn matplotlib seaborn tensorflow
```
 
---
 
## 🚀 How to Run
 
1. Clone the repository and place `titanic.csv` in the `data/` folder
2. Open `Fahad_ipynb1.ipynb` in Jupyter or VS Code
3. Run all cells in order
---
 
## 🎓 Acknowledgments
 
Special thanks to **Engineer Miral** for her outstanding guidance and continuous support throughout this project.
