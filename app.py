import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =========================================
# PAGE TITLE
# =========================================

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Marks Analysis & Grade Predictor")

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("StudentsPerformance.csv")

# Create Average Score
df['average score'] = df[[
    'math score',
    'reading score',
    'writing score'
]].mean(axis=1)

# =========================================
# DATA PREVIEW
# =========================================

st.header("📊 Dataset Preview")

st.dataframe(df.head())

# =========================================
# VISUALIZATIONS
# =========================================

st.header("📈 Visualizations")

# Histogram
fig1, ax1 = plt.subplots(figsize=(8,5))

sns.histplot(
    df['average score'],
    bins=20,
    kde=True,
    color='skyblue',
    ax=ax1
)

ax1.set_title("Distribution of Average Scores")

st.pyplot(fig1)

# Scatter Plot
fig2, ax2 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x='math score',
    y='reading score',
    hue='gender',
    data=df,
    ax=ax2
)

ax2.set_title("Math Score vs Reading Score")

st.pyplot(fig2)

# Heatmap
fig3, ax3 = plt.subplots(figsize=(6,4))

corr = df[[
    'math score',
    'reading score',
    'writing score',
    'average score'
]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    ax=ax3
)

ax3.set_title("Correlation Heatmap")

st.pyplot(fig3)

# =========================================
# MACHINE LEARNING MODEL
# =========================================

X = df[['reading score', 'writing score']]
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

# =========================================
# GRADE FUNCTION
# =========================================

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

# =========================================
# USER INPUT
# =========================================

st.header("🤖 Predict Student Grade")

reading = st.slider(
    "Select Reading Score",
    0,
    100,
    50
)

writing = st.slider(
    "Select Writing Score",
    0,
    100,
    50
)

# Prediction Button
if st.button("Predict"):

    prediction = model.predict([[reading, writing]])

    predicted_math = prediction[0]

    average = (
        predicted_math + reading + writing
    ) / 3

    grade = get_grade(average)

    st.success(
        f"Predicted Math Score : {predicted_math:.2f}"
    )

    st.info(
        f"Predicted Average Score : {average:.2f}"
    )

    st.warning(
        f"Predicted Grade : {grade}"
    )

# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.markdown("Developed using Python, Streamlit & Machine Learning")