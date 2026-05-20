import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

# Set plotting style and ignore warnings for cleaner output
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')

def main():
    print("--- Student Marks Analysis & Grade Predictor ---\n")
    
    # ---------------------------------------------------------
    # 1. Data Loading 
    # ---------------------------------------------------------
    file_path = 'StudentsPerformance.csv'
    
   
    # Load the dataset
    df = pd.read_csv("StudentsPerformance.csv")
    
    
    # Create an average score column for general evaluation
    df['average score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    
    print("Data Overview:")
    print(df.head(), "\n")
    
    # ---------------------------------------------------------
    # 2. Exploratory Data Analysis & Visualizations
    # ---------------------------------------------------------
    print("Generating visualizations (Close each window to proceed to the next)...")
    
    # Visualization 1: Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df['average score'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Average Scores')
    plt.xlabel('Average Score')
    plt.ylabel('Frequency')
    plt.show()
    
    # Visualization 2: Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='math score', y='reading score', hue='gender', data=df, alpha=0.7)
    plt.title('Math Score vs Reading Score by Gender')
    plt.xlabel('Math Score')
    plt.ylabel('Reading Score')
    plt.show()
    
    # Visualization 3: Heatmap (Correlation)
    plt.figure(figsize=(6, 4))
    numerical_cols = df[['math score', 'reading score', 'writing score', 'average score']]
    corr_matrix = numerical_cols.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Student Scores')
    plt.show()

    # ---------------------------------------------------------
    # 3. Grade Predictor Model using Linear Regression
    # ---------------------------------------------------------
    print("\n--- Training Grade Predictor (Linear Regression) ---")
    
    # Data Preprocessing: Convert categorical variables into dummy/indicator variables
    features = df.drop(columns=['math score', 'reading score', 'writing score', 'average score'])
    target = df['average score']
    
    # One-hot encoding for categorical data
    X = pd.get_dummies(features, drop_first=True)
    y = target
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples\n")
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared Score (R2): {r2:.2f}\n")
    
    # Grade prediction mapping
    def get_grade(score):
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    # Compare actual vs predicted grades for the first 10 students in the test set
    results_df = pd.DataFrame({
        'Actual Score': y_test.values,
        'Predicted Score': y_pred,
    })
    
    results_df['Actual Grade'] = results_df['Actual Score'].apply(get_grade)
    results_df['Predicted Grade'] = results_df['Predicted Score'].apply(get_grade)
    
    print("Grade Prediction Results (First 10 Students):")
    print(results_df.head(10))

if __name__ == "__main__":
    main()
