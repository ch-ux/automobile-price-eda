import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Page configuration
st.set_page_config(
    page_title="Automobile Price EDA",
    layout="wide"
)

st.title("🚗 Automobile Price Data – Advanced Exploratory Data Analysis")

# =========================
# Load & Clean Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("df_0.csv", na_values="?")

    # Convert numeric-looking object columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
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
# Column Groups
# =========================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# =========================
# Distribution Analysis
# =========================
st.subheader("📊 Univariate Distribution Analysis")
selected_numeric = st.selectbox("Select Numeric Column", numeric_cols)

fig, ax = plt.subplots()
sns.histplot(df[selected_numeric], kde=True, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x=df[selected_numeric], ax=ax)
st.pyplot(fig)

# =========================
# Price vs Numeric (Regression + Correlation)
# =========================
if "price" in df.columns:
    st.subheader("📈 Price vs Numeric Feature Analysis")

    price_numeric = [col for col in numeric_cols if col != "price"]
    selected_feature = st.selectbox("Select Feature", price_numeric)

    # Regression plot
    fig, ax = plt.subplots()
    sns.regplot(x=df[selected_feature], y=df["price"], ax=ax, scatter_kws={"alpha": 0.5})
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # Pearson correlation & p-value
    clean_df = df[[selected_feature, "price"]].dropna()

    if len(clean_df) > 2:
        corr, p_val = pearsonr(clean_df[selected_feature], clean_df["price"])
        st.markdown("### 📐 Statistical Significance")
        st.write(f"**Pearson Correlation Coefficient:** {corr:.4f}")
        st.write(f"**P-value:** {p_val:.4e}")

        if p_val < 0.05:
            st.success("Statistically significant relationship (p < 0.05)")
        else:
            st.warning("Not statistically significant (p ≥ 0.05)")

# =========================
# Correlation Matrix with Price Focus
# =========================
if "price" in df.columns:
    st.subheader("🔗 Correlation of Features with Price")

    corr_matrix = df[numeric_cols].corr()
    price_corr = corr_matrix[["price"]].sort_values(by="price", ascending=False)

    st.dataframe(price_corr)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(price_corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =========================
# Multivariate Pairplot (Top Correlated Features)
# =========================
if "price" in df.columns:
    st.subheader("📊 Pairwise Relationships (Top Correlated Features)")

    top_features = (
        df[numeric_cols]
        .corr()["price"]
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    pairplot_df = df[list(top_features)].dropna()
    fig = sns.pairplot(pairplot_df, diag_kind="kde")
    st.pyplot(fig)
