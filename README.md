# Student Marks Analysis & Grade Predictor

## Overview

The Student Marks Analysis & Grade Predictor is a machine learning-based mini project developed using Python.
This project analyzes student performance data, performs data cleaning and visualization, and predicts student math scores and grades based on reading and writing scores.

The project uses Linear Regression for prediction and includes various Exploratory Data Analysis (EDA) visualizations.

---

## Features

* Dataset preprocessing and cleaning
* Missing value handling
* Duplicate record removal
* Categorical data encoding
* Exploratory Data Analysis (EDA)
* Grade prediction system
* Linear Regression model training
* User input prediction system
* Visualization using Matplotlib and Seaborn

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn & streamlit

---

## Dataset

Dataset Used:

* Students Performance Dataset

Dataset File:

* `StudentsPerformance.csv`

Dataset contains:

* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch Type
* Test Preparation Course
* Math Score
* Reading Score
* Writing Score

---

## Project Structure

```bash
Student_Marks_Analysis/
│__ app.py
├── student_analysis.py
├── StudentsPerformance.csv
├── README.md
└── requirements.txt
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone <your-github-repository-link>
```

### Step 2: Open Project Folder

```bash
cd Student_Marks_Analysis
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run the Project

Run the project using:

```bash
python student_analysis.py        
     or 
streamlit run app.py
```
---

## Machine Learning Model

Algorithm Used:

* Linear Regression

Input Features:

* Reading Score
* Writing Score

Target Variable:

* Math Score

---

## Visualizations Included

* Histogram
* Scatter Plot
* Correlation Heatmap
* Grade Distribution Graph
* Actual vs Predicted Score Graph

---

## Grade Classification

| Average Score | Grade |
| ------------- | ----- |
| 90 and above  | A     |
| 75 - 89       | B     |
| 60 - 74       | C     |
| 40 - 59       | D     |
| Below 40      | F     |

---

## Output

The system predicts:

* Math Score
* Average Score
* Student Grade

based on user-entered reading and writing scores.

---

## Future Enhancements

* Add Streamlit Web Interface
* Improve prediction accuracy
* Use advanced ML algorithms
* Deploy project online
* Add student performance dashboard

---

## Author

Sharanabasappa
BTech(CSE) - AI & ML
Rai Technology University

---

## License

This project is developed for educational and academic purposes.
