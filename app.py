import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

st.set_page_config(page_title="Regresi Berganda", layout="wide")

# ---------- HEADER ----------
st.title("📊 Aplikasi Analisis Regresi Berganda")
st.markdown("""
Unggah file CSV yang berisi data numerik, lalu pilih variabel dependen dan independennya.
Aplikasi ini juga menyertakan uji asumsi dan visualisasi hasil regresi secara lengkap.
""")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("📥 Upload File CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Pratinjau Data")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    dep_var = st.sidebar.selectbox("🎯 Pilih Variabel Dependen (Y)", numeric_cols)
    indep_vars = st.sidebar.multiselect("📈 Pilih Variabel Independen (X)", [col for col in numeric_cols if col != dep_var])

    if dep_var and indep_vars:
        X = df[indep_vars]
        y = df[dep_var]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        st.subheader("📋 Hasil Regresi")
        st.text(model.summary())

        st.subheader("📊 Visualisasi Diagnostik")

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax1,
                          line_kws={'color': 'red', 'lw': 1})
            ax1.set_xlabel("Fitted Values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residual vs Fitted")
            st.pyplot(fig1)

        with col2:
            fig2 = sm.qqplot(model.resid, line='45')
            plt.title("QQ Plot")
            st.pyplot(fig2.figure)

        col3, col4 = st.columns(2)
        with col3:
            fig3, ax3 = plt.subplots()
            sns.histplot(model.resid, kde=True, ax=ax3)
            ax3.set_title("Distribusi Residual")
            st.pyplot(fig3)

        with col4:
            fig4, ax4 = plt.subplots()
            sm.graphics.influence_plot(model, ax=ax4, criterion="cooks")
            st.pyplot(fig4)

        # ---------- Uji Asumsi ----------
        st.subheader("🧪 Uji Asumsi Regresi Berganda")

        st.markdown("### 1. Uji Normalitas Residual (Shapiro-Wilk)")
        stat, p = stats.shapiro(model.resid)
        st.write(f"**Statistic** = {stat:.4f}, **p-value** = {p:.4f}")
        if p > 0.05:
            st.success("Residual terdistribusi normal (p > 0.05)")
        else:
            st.error("Residual tidak terdistribusi normal (p ≤ 0.05)")

        st.markdown("### 2. Uji Homoskedastisitas (Breusch-Pagan)")
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        bp_labels = ['LM Statistic', 'LM p-value', 'F-Statistic', 'F p-value']
        for label, value in zip(bp_labels, bp_test):
            st.write(f"**{label}**: {value:.4f}")
        if bp_test[3] > 0.05:
            st.success("Homoskedastisitas terpenuhi (p > 0.05)")
        else:
            st.error("Terdapat heteroskedastisitas (p ≤ 0.05)")

        st.markdown("### 3. Uji Autokorelasi (Durbin-Watson)")
        dw = durbin_watson(model.resid)
        st.write(f"**Durbin-Watson**: {dw:.4f}")
        if 1.5 < dw < 2.5:
            st.success("Tidak ada autokorelasi yang kuat (nilai mendekati 2)")
        else:
            st.error("Ada indikasi autokorelasi (nilai jauh dari 2)")

        st.markdown("### 4. Uji Multikolinearitas (VIF)")
        vif_df = pd.DataFrame()
        vif_df["Variabel"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_df)

    else:
        st.warning("⚠️ Silakan pilih variabel dependen dan minimal satu variabel independen.")
else:
    st.info("💡 Upload file CSV terlebih dahulu untuk memulai.")
