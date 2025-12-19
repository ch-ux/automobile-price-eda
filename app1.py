import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Automobile Price EDA",
    layout="wide"
)

st.title("🚗 Automobile Price Data – Exploratory Data Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("df_0.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.header("EDA Controls")

# =========================
# Dataset Overview
# =========================
st.subheader("📌 Dataset Overview")
st.write("Shape of dataset:", df.shape)
st.dataframe(df.head())

# =========================
# Data Types
# =========================
st.subheader("📌 Data Types")
st.write(df.dtypes)

# =========================
# Missing Values
# =========================
st.subheader("📌 Missing Values Analysis")
missing = df.isnull().sum()
st.write(missing[missing > 0])

# =========================
# Summary Statistics
# =========================
st.subheader("📌 Summary Statistics")
st.write(df.describe(include="all"))

# =========================
# Column Selection
# =========================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

selected_numeric = st.sidebar.selectbox(
    "Select Numeric Column",
    numeric_cols
)

selected_categorical = st.sidebar.selectbox(
    "Select Categorical Column",
    categorical_cols
)

# =========================
# Distribution Plot
# =========================
st.subheader(f"📊 Distribution of {selected_numeric}")
fig, ax = plt.subplots()
sns.histplot(df[selected_numeric], kde=True, ax=ax)
st.pyplot(fig)

# =========================
# Boxplot (Outliers)
# =========================
st.subheader(f"📦 Boxplot of {selected_numeric}")
fig, ax = plt.subplots()
sns.boxplot(x=df[selected_numeric], ax=ax)
st.pyplot(fig)

# =========================
# Categorical vs Price Analysis
# =========================
if "price" in df.columns:
    st.subheader(f"💰 Price vs {selected_categorical}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df[selected_categorical], y=df["price"], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =========================
# Correlation Heatmap
# =========================
st.subheader("📈 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)
st.pyplot(fig)
