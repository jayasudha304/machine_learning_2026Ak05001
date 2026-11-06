# BITS Pilani - M.Tech in AIML (WILP)
## Course: Machine Learning | Assignment - 2
### **Student ID: 2026Ak05001**

---

## a. Problem Statement
The objective of this project is to develop, evaluate, and deploy an end-to-end Machine Learning classification pipeline and interactive Streamlit web application to predict whether a client will subscribe to a bank term deposit (`deposit`: binary classification, `0` = No, `1` = Yes) based on direct marketing campaigns.

Direct marketing campaigns via telephone represent a significant operational cost for banking institutions. Building a robust predictive model allows financial institutions to identify high-potential customers, optimize tele-marketing resource allocation, reduce operational overhead, and improve campaign conversion rates. In accordance with the assignment guidelines, five distinct classification algorithms are trained on the same dataset, benchmarked across six standard performance metrics, serialized, and deployed on **Streamlit Community Cloud** with interactive evaluation capabilities without retraining.

---

## b. Dataset Description

* **Dataset Selected:** [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **Public Repository Source:** UCI Machine Learning Repository & Kaggle (S. Moro, P. Cortez, and P. Rita, 2014)
* **Instance Count:** **4,521 instances** (Strictly satisfies the minimum instance requirement of $\ge 500$)
* **Feature Count:** **16 input features** (Strictly satisfies the minimum feature requirement of $\ge 12$)
* **Task Type:** Binary Classification
* **Target Variable:** `deposit` (`0` = Client did not subscribe, `1` = Client subscribed to a term deposit)
* **Class Distribution:** ~84.6% Class 0 (No) and ~15.4% Class 1 (Yes) — reflecting real-world bank campaign imbalance.

### Feature Breakdown:
1. **Numerical Input Features (7):**
   * `age`: Age of customer (Numeric, Years: 18 – 90)
   * `balance`: Average yearly bank balance (Numeric, in Euros)
   * `day`: Last contact day of the month (Numeric: 1 – 31)
   * `duration`: Last contact duration in seconds (Numeric: 5 – 3,000 s)
   * `campaign`: Number of contacts performed during this campaign for this client (Numeric: 1 – 30)
   * `pdays`: Number of days passed after client was last contacted from a previous campaign (Numeric, `-1` = client not previously contacted)
   * `previous`: Number of contacts performed before this campaign (Numeric: 0 – 15)

2. **Categorical Input Features (9):**
   * `job`: Type of employment (`admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`)
   * `marital`: Marital status (`married`, `single`, `divorced`)
   * `education`: Education level (`primary`, `secondary`, `tertiary`, `unknown`)
   * `default`: Has credit in default? (`no`, `yes`)
   * `housing`: Has housing loan? (`no`, `yes`)
   * `loan`: Has personal loan? (`no`, `yes`)
   * `contact`: Contact communication type (`cellular`, `telephone`, `unknown`)
   * `month`: Last contact month of year (`jan`, `feb`, `mar`, ..., `nov`, `dec`)
   * `poutcome`: Outcome of the previous marketing campaign (`unknown`, `failure`, `other`, `success`)

3. **Fixed Test Dataset:**
   * An 80–20 stratified train-test split was performed using `random_state=42`.
   * The fixed holdout test dataset comprises **905 instances** and is saved in `data/test_data.csv`.
   * It is automatically pre-loaded on Streamlit startup by default.

---

## c. Github Repository Link

* **GitHub Repository URL:** [https://github.com/jayasudha304/machine_learning_2026Ak05001.git](https://github.com/jayasudha304/machine_learning_2026Ak05001.git)
* **Streamlit Community Cloud Live App:** [https://machinelearning2026ak05001-8ngvhp8jxbd4gbvgnxvdwf.streamlit.app/](https://machinelearning2026ak05001-8ngvhp8jxbd4gbvgnxvdwf.streamlit.app/)

*The repository contains all required files for full reproducibility: `app.py`, `train.py`, `requirements.txt`, `README.md`, `data/` (including `test_data.csv`), and serialized model pipelines in `models/`.*

---

## d. Models Used

All five models are built using a unified Scikit-Learn `Pipeline` with a `ColumnTransformer` preprocessor (`StandardScaler` for numeric features and `OneHotEncoder(handle_unknown='ignore')` for categorical features):
1. **Logistic Regression:** L-BFGS solver, `max_iter=1000`, `C=1.0`, `random_state=42`
2. **Decision Tree Classifier:** Gini impurity, `max_depth=6`, `min_samples_leaf=10`, `random_state=42`
3. **K-Nearest Neighbor Classifier (KNN):** `n_neighbors=7`, Minkowski metric ($p=2$), uniform weights
4. **Naive Bayes Classifier:** Gaussian Naive Bayes (`GaussianNB()`)
5. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel, $C=1.0$, `probability=True`, `random_state=42`

### Comparison Table: Evaluation Metrics Calculated for All 5 Models

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8950** | **0.8850** | **0.7200** | **0.5100** | **0.5970** | **0.5510** |
| **Decision Tree** | **0.8900** | **0.8500** | **0.6800** | **0.5300** | **0.5960** | **0.5420** |
| **KNN** | **0.8800** | **0.8350** | **0.6500** | **0.4500** | **0.5320** | **0.4780** |
| **Naive Bayes** | **0.8350** | **0.8400** | **0.4700** | **0.6800** | **0.5560** | **0.4850** |
| **Support Vector Machine** | **0.9020** | **0.8920** | **0.7600** | **0.5200** | **0.6180** | **0.5780** |

---

### Observations on the Performance of Each Model on the Chosen Dataset

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a strong, computationally efficient linear baseline. When paired with standard scaling and one-hot encoding, it achieves solid overall accuracy (0.8950) and a high AUC (0.8850). It effectively weights continuous features like call duration and financial balance, but exhibits moderate recall (0.5100) on the minority deposit class due to its inability to capture non-linear feature interactions. |
| **Decision Tree** | Effectively models non-linear interactions between demographic indicators (job, housing loan) and campaign attributes (poutcome, month). Constraining tree complexity with `max_depth=6` and `min_samples_leaf=10` prevents overfitting, yielding balanced precision (0.6800) and recall (0.5300) with an MCC of 0.5420. It provides transparent, human-interpretable decision boundaries. |
| **KNN** | As an instance-based non-parametric classifier, KNN achieves reasonable accuracy (0.8800) but suffers from the curse of dimensionality across the one-hot encoded categorical space (expanded sparse feature dimensions). This leads to distorted Euclidean distances and the lowest recall (0.4500) and lowest MCC (0.4780) among the models, alongside higher inference latency. |
| **Naive Bayes** | Assumes conditional feature independence given the class label. While this assumption is violated by correlated features (such as duration, campaign contacts, and pdays), GaussianNB achieves the **highest recall (0.6800)** on the minority class (`deposit = 1`). However, this comes at the cost of the lowest precision (0.4700) and lowest accuracy (0.8350) due to elevated false positive alarms. Ideal when missing a prospective subscriber is deemed more costly than a false lead. |
| **Support Vector Machine** | **Top-performing model overall.** By leveraging the non-linear Radial Basis Function (RBF) kernel with Platt scaling (`probability=True`), SVM maps the 16 features into a high-dimensional space, isolating optimal non-linear separating hyperplanes with maximum margins. It achieves the highest Accuracy (0.9020), highest AUC (0.8920), highest Precision (0.7600), highest F1-Score (0.6180), and highest MCC (0.5780). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM with RBF Kernel)** is the undisputed champion model on this dataset. It secures the highest rank across 5 out of 6 evaluation metrics (Accuracy: 0.9020, AUC: 0.8920, Precision: 0.7600, F1-Score: 0.6180, and MCC: 0.5780). Its superior Matthews Correlation Coefficient (MCC = 0.5780) confirms exceptional diagnostic balance on this class-imbalanced banking dataset. |

---

## e. Streamlit Web Application Features & Cloud Optimization

1. **Default Test Data Pre-loaded:** Loads `data/test_data.csv` (905 rows, 16 features) on startup.
2. **Student ID Display:** BITS ID `2026Ak05001` displayed in both sidebar and main header.
3. **Dataset Differentiation:** Enlarged visual banners (Blue for Default, Amber for Custom Upload).
4. **Strict Schema & $\ge 12$ Features Validation:** Rejects unexpected/invalid uploads without loading default dataset; displays an error screen with an enlarged clear button.
5. **No Retraining:** Pre-trained `.joblib` pipelines evaluate uploaded data with zero retraining.
6. **All Models Comparative View:** Displays 6-metric leaderboard, champion KPI cards, bar charts, ROC curves, side-by-side Confusion Matrices, and Classification Reports.
7. **Cloud Memory & Cache Manager:** Expander with cache-clearing button (`st.cache_data.clear()`, `st.cache_resource.clear()`, `plt.close('all')`, `gc.collect()`) to prevent OOM errors on Streamlit Community Cloud.

---

## f. Local Execution Guide (Windows / VS Code)

1. Open project folder in VS Code:
   ```powershell
   cd bits_aiml_assignment2
   ```
2. Create and activate virtual environment:
   ```powershell
   python -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Train models and generate pre-saved pipelines:
   ```powershell
   python train.py
   ```
5. Launch the Streamlit application:
   ```powershell
   streamlit run app.py
   ```
