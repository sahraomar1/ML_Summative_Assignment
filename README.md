# Predicting Labour Force Status in Rwanda: Traditional Machine Learning vs. Deep Learning

This repository contains machine learning and deep learning models that predict individual employment status using 66,303 records from the national Rwanda Labour Force Survey.

The goal is to classify each person into one of three categories:
* **Employed** (50.67% of the data)
* **Out of Labour Force** (40.51% of the data)
* **Unemployed** (8.82% of the data)

The main challenge is class imbalance. Because less than 9% of respondents are unemployed, basic models tend to ignore this group to get a higher overall accuracy. This project examines how cost-sensitive class weighting and different model types deal with this problem.

---

## Project Links
* **GitHub Repository:** [https://github.com/sahraomar1/ML_Summative_Assignment]
* **Google Drive Folder (Data and Saved Models):** [https://drive.google.com/drive/folders/1LAvPV0CvnoiiVHBxzqwZ71Wv5uhCfVsU?usp=drive_link]

---

## Model Benchmark Results

All five models were tested on the same hold-out test set of 13,261 records, which contains 1,170 unemployed people.

| Model | Overall Accuracy | Macro F1-Score | Unemployed Recall | Type |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline Logistic Regression** | 59.77% | 0.4124 | 0.00% | Traditional ML |
| **Balanced Logistic Regression** | 51.53% | 0.4472 | 33.93% | Traditional ML |
| **Random Forest (Balanced)** | **62.56%** | **0.5141** | 22.82% | Traditional ML |
| **XGBoost (Cost-Sensitive)** | 56.77% | 0.5034 | **45.13%** | Traditional ML |
| **Deep Learning MLP (`tf.data`)** | 58.76% | 0.5109 | 39.06% | Deep Learning |

### Key Findings
* **The unweighted baseline failed on the minority class:** The basic Logistic Regression reached 59.77% accuracy by predicting only the two large classes. It found 0 unemployed people.
* **Class weights restored detection:** Adding class weights lowered accuracy to 51.53%, but raised unemployed recall from 0% to 33.93% (397 job-seekers found).
* **Random Forest had the best overall balance:** It had the highest accuracy (62.56%) and the highest Macro F1-score (0.5141).
* **XGBoost found the most unemployed people:** Its recall on the small class was 45.13% (528 job-seekers found).
* **The Deep Learning MLP was competitive:** Built with `tf.data`, batch normalization, and dropout, it reached 58.76% accuracy, 0.5109 Macro F1, and 39.06% unemployed recall.
* **Main drivers:** Feature importance showed that age (44.52%) and household size (19.61%) make up nearly two-thirds of the Random Forest's total importance.

---

## Repository Structure

```text
├── figures/
│   ├── eda/                          # Plots of the target and demographic features
│   ├── confusion_matrices/           # Normalized confusion matrices for all models
│   ├── roc_pr_curves/                # One-vs-rest ROC curves for Random Forest
│   ├── dl_learning_curves.png        # Training vs. validation loss and accuracy curves
│   └── rf_feature_importance.png     # Top 10 feature importance bar chart
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_pipeline_and_modeling.ipynb
├── .gitignore                        # Ignores local data/ and models/ folders
├── README.md                         # Project documentation and results overview
└── requirements.txt                  # Python dependencies
Note: The data/ and models/ folders are not stored on GitHub. They are available in the Google Drive folder linked above.

How to Run This Project
1. Clone the repository
Bash
git clone [https://github.com/sahraomar1/ML_Summative_Assignment.git]
cd ML_Summative_Assignment
2. Create and activate a virtual environment
Bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
3. Install the dependencies
Bash
pip install -r requirements.txt
4. Add the data
Download the data from the Google Drive folder linked above. Create a data/ folder in the project root and place the downloaded data file(s) inside it. Optionally, place the saved model files in a models/ folder in the project root.

5. Run the notebooks
Run them in this order:

Open notebooks/01_exploratory_data_analysis.ipynb for data cleaning, distribution checks, and exploratory plots.

Open notebooks/02_pipeline_and_modeling.ipynb for the preprocessing pipeline, model training, evaluation, and charts.

The notebooks save their plots directly to the figures/ folder.