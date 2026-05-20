# =========================================================
# STUDENT MARKS ANALYSIS & GRADE PREDICTOR
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set graph style
sns.set_theme(style="whitegrid")


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    print("\n========== STUDENT MARKS ANALYSIS ==========\n")

    # =====================================================
    # 1. LOAD DATASET
    # =====================================================

    # Load CSV file
    df = pd.read_csv("StudentsPerformance.csv")

    # Create Average Score Column
    df['average score'] = df[
        ['math score', 'reading score', 'writing score']
    ].mean(axis=1)

    # =====================================================
    # 2. DATA PREVIEW
    # =====================================================

    print("Dataset Preview:\n")
    print(df.head())

    print("\nDataset Information:\n")
    print(df.info())

    print("\nStatistical Summary:\n")
    print(df.describe())

    # =====================================================
    # 3. VISUALIZATIONS
    # =====================================================

    print("\nGenerating Visualizations...\n")

    # -----------------------------------------------------
    # VISUALIZATION 1 : HISTOGRAM
    # -----------------------------------------------------

    plt.figure(figsize=(8,5))

    sns.histplot(
        df['average score'],
        bins=20,
        kde=True,
        color='skyblue'
    )

    plt.title('Distribution of Average Scores')
    plt.xlabel('Average Score')
    plt.ylabel('Frequency')

    plt.show()

    # -----------------------------------------------------
    # VISUALIZATION 2 : SCATTER PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        x='math score',
        y='reading score',
        hue='gender',
        data=df,
        alpha=0.7
    )

    plt.title('Math Score vs Reading Score')
    plt.xlabel('Math Score')
    plt.ylabel('Reading Score')

    plt.show()

    # -----------------------------------------------------
    # VISUALIZATION 3 : HEATMAP
    # -----------------------------------------------------

    plt.figure(figsize=(6,4))

    numerical_cols = df[
        ['math score', 'reading score', 'writing score', 'average score']
    ]

    corr_matrix = numerical_cols.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5
    )

    plt.title('Correlation Heatmap')

    plt.show()

    # -----------------------------------------------------
    # BONUS VISUALIZATION : GRADE DISTRIBUTION
    # -----------------------------------------------------

    def get_grade(score):

        if score >= 90:
            return 'A'

        elif score >= 75:
            return 'B'

        elif score >= 60:
            return 'C'

        elif score >= 40:
            return 'D'

        else:
            return 'F'

    # Create Grade Column
    df['grade'] = df['average score'].apply(get_grade)

    plt.figure(figsize=(7,5))

    sns.countplot(
        x='grade',
        data=df,
        palette='viridis',
        order=['A', 'B', 'C', 'D', 'F']
    )

    plt.title('Student Grade Distribution')
    plt.xlabel('Grade')
    plt.ylabel('Number of Students')

    plt.show()

    # =====================================================
    # 4. MACHINE LEARNING MODEL
    # =====================================================

    print("\n========== TRAINING MODEL ==========\n")

    # Features (Input)
    X = df[['reading score', 'writing score']]

    # Target (Output)
    y = df['math score']

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")

    # Create Linear Regression Model
    model = LinearRegression()

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # =====================================================
    # 5. MODEL EVALUATION
    # =====================================================

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:\n")

    print(f"Mean Squared Error : {mse:.2f}")
    print(f"R2 Score           : {r2:.2f}")

    # =====================================================
    # 6. ACTUAL VS PREDICTED GRAPH
    # =====================================================

    plt.figure(figsize=(8,5))

    plt.scatter(y_test, y_pred)

    plt.xlabel("Actual Math Scores")
    plt.ylabel("Predicted Math Scores")

    plt.title("Actual vs Predicted Scores")

    plt.show()

    # =====================================================
    # 7. PREDICTION RESULTS TABLE
    # =====================================================

    results_df = pd.DataFrame({

        'Actual Score': y_test.values,
        'Predicted Score': y_pred

    })

    results_df['Actual Grade'] = results_df[
        'Actual Score'
    ].apply(get_grade)

    results_df['Predicted Grade'] = results_df[
        'Predicted Score'
    ].apply(get_grade)

    print("\n========== PREDICTION RESULTS ==========\n")

    print(results_df.head(10))

    # =====================================================
    # 8. USER INPUT PREDICTION LOOP
    # =====================================================

    print("\n========== STUDENT SCORE PREDICTION ==========\n")

    print("Type 'exit' anytime to stop prediction.\n")

    while True:

        # Reading Score Input
        reading_input = input("Enter Reading Score : ")

        # Exit Condition
        if reading_input.lower() == "exit":
            print("\nPrediction system closed.")
            break

        # Writing Score Input
        writing_input = input("Enter Writing Score : ")

        # Exit Condition
        if writing_input.lower() == "exit":
            print("\nPrediction system closed.")
            break
        try:
            # Validate Input
            reading = float(reading_input)
            writing = float(writing_input)

            if not (0 <= reading <= 100) or not (0 <= writing <= 100):
                print("Please enter scores between 0 and 100.")
                continue
        except ValueError:
            print("Please enter valid numeric values for the scores.")
            continue

        # Convert input to float
        reading = float(reading_input)
        writing = float(writing_input)

        # Predict Math Score
        prediction = model.predict([[reading, writing]])

        predicted_math = prediction[0]

        print(f"\nPredicted Math Score : {predicted_math:.2f}")

        # Calculate Average Score
        average = (predicted_math + reading + writing) / 3

        print(f"Predicted Average Score : {average:.2f}")

        # Predict Grade
        predicted_grade = get_grade(average)

        print(f"Predicted Grade : {predicted_grade}")

        print("\n-----------------------------------\n")

    print("\n========== PROJECT COMPLETED ==========\n")


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()