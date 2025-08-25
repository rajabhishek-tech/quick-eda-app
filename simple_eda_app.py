# simple_eda_app.py
# Instant EDA: upload a CSV/XLSX and get automatic analysis.

import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quick EDA", layout="wide")
st.title("Quick EDA — Upload & Analyze")

st.caption(
    "Upload a CSV or Excel file. This app profiles columns, shows unique counts, "
    "missing data, basic statistics, top value counts, and draws charts automatically."
)

def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 ** 2)

def split_columns(df: pd.DataFrame, cat_max_uniq: int = 30) -> Tuple[List[str], List[str], List[str]]:
    num_cols, cat_cols, dt_cols = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            nunique = s.nunique(dropna=True)
            try:
                pd.to_datetime(s, errors="raise")
                dt_cols.append(c)
                continue
            except Exception:
                pass
            if nunique <= cat_max_uniq:
                cat_cols.append(c)
            else:
                cat_cols.append(c)
    num_cols = [c for c in num_cols if c not in dt_cols]
    cat_cols = [c for c in cat_cols if c not in dt_cols]
    return num_cols, cat_cols, dt_cols

def safe_sample(df: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    if len(df) > n:
        return df.sample(n, random_state=42)
    return df

def draw_hist(series: pd.Series, bins: int = 30, title: str = ""):
    plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title or f"Histogram: {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.tight_layout()
    return plt.gcf()

def draw_box(series: pd.Series, title: str = ""):
    plt.figure()
    plt.boxplot(series.dropna().values, vert=True, labels=[series.name])
    plt.title(title or f"Boxplot: {series.name}")
    plt.tight_layout()
    return plt.gcf()

def draw_corr_heatmap(df_num: pd.DataFrame):
    corr = df_num.corr(numeric_only=True)
    if corr.empty:
        return None
    plt.figure(figsize=(min(10, 0.6*len(corr.columns)+2), min(10, 0.6*len(corr.columns)+2)))
    im = plt.imshow(corr, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = range(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=60, ha="right")
    plt.yticks(ticks, corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return plt.gcf()

def top_value_counts(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    vc = series.astype("string").fillna("<NA>").value_counts().head(top_n)
    return vc.rename_axis("value").reset_index(name="count")

with st.sidebar:
    st.header("Options")
    sep = st.text_input("CSV separator (ignored for Excel)", value=",")
    max_rows_preview = st.slider("Rows to preview", 5, 200, 20, step=5)

uploaded = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded, sep=sep)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(f"Loaded **{uploaded.name}** — rows: {len(df):,}, columns: {len(df.columns)}")
with st.expander("Preview (head)"):
    st.dataframe(df.head(max_rows_preview))

st.header("Overview")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a: st.metric("Rows", f"{len(df):,}")
with col_b: st.metric("Columns", f"{len(df.columns):,}")
with col_c: st.metric("Duplicate rows", f"{df.duplicated().sum():,}")
with col_d: st.metric("Memory (MB)", f"{memory_usage_mb(df):.2f}")

dtype_tbl = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "non_null": [df[c].notna().sum() for c in df.columns],
    "missing": [df[c].isna().sum() for c in df.columns],
})
dtype_tbl["missing_%"] = (dtype_tbl["missing"] / len(df) * 100).round(2)
dtype_tbl["unique"] = [df[c].nunique(dropna=True) for c in df.columns]
st.subheader("Column types & health")
st.dataframe(dtype_tbl.sort_values(["dtype", "column"]).reset_index(drop=True))

with st.expander("Missing values bar (top 30)"):
    miss_sorted = dtype_tbl.sort_values("missing", ascending=False).head(30)
    if miss_sorted["missing"].sum() == 0:
        st.write("No missing values detected.")
    else:
        plt.figure()
        plt.bar(miss_sorted["column"], miss_sorted["missing"])
        plt.xticks(rotation=60, ha="right")
        plt.title("Missing values per column (top 30)")
        plt.tight_layout()
        st.pyplot(plt.gcf())

num_cols, cat_cols, dt_cols = split_columns(df)
c1, c2, c3 = st.columns(3)
with c1: st.write("**Numeric columns:**", num_cols if num_cols else "—")
with c2: st.write("**Categorical columns:**", cat_cols if cat_cols else "—")
with c3: st.write("**Datetime-like columns:**", dt_cols if dt_cols else "—")

if num_cols:
    st.header("Numeric — Statistics")
    st.dataframe(df[num_cols].describe().T)

    st.subheader("Numeric — Histograms")
    sample_df = safe_sample(df[num_cols], 50_000)
    grid = st.columns(2)
    for i, c in enumerate(num_cols):
        fig = draw_hist(sample_df[c], title=f"Histogram: {c}")
        with grid[i % 2]:
            st.pyplot(fig)

    st.subheader("Numeric — Boxplots")
    grid2 = st.columns(2)
    for i, c in enumerate(num_cols):
        fig = draw_box(sample_df[c], title=f"Boxplot: {c}")
        with grid2[i % 2]:
            st.pyplot(fig)

    if len(num_cols) >= 2:
        st.subheader("Correlation Heatmap")
        fig = draw_corr_heatmap(safe_sample(df[num_cols], 50_000))
        if fig is not None:
            st.pyplot(fig)
        else:
            st.write("Not enough numeric data for correlation.")
else:
    st.info("No numeric columns detected.")

if cat_cols:
    st.header("Categorical — Top value counts")
    top_n = st.slider("Top N per categorical column", 5, 30, 10, step=1)
    for c in cat_cols:
        st.markdown(f"**{c}**")
        counts = top_value_counts(df[c], top_n)
        st.dataframe(counts)
        plt.figure()
        plt.bar(counts["value"].astype(str), counts["count"].values)
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        st.pyplot(plt.gcf())
else:
    st.info("No categorical columns detected.")

if dt_cols:
    st.header("Datetime — Breakdown")
    for c in dt_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in dt_cols:
        st.markdown(f"**{c}**")
        s = df[c].dropna()
        if s.empty:
            st.write("No valid datetime values.")
            continue
        g = s.dt.to_period("M").value_counts().sort_index().rename_axis("month").reset_index(name="count")
        g["month"] = g["month"].astype(str)
        st.dataframe(g.tail(24))
        plt.figure()
        plt.plot(g["month"], g["count"])
        plt.xticks(rotation=60, ha="right")
        plt.title(f"Monthly count — {c}")
        plt.tight_layout()
        st.pyplot(plt.gcf())
