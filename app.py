import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import os

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="Prediksi Status Akademik",
    page_icon="🎓",
    layout="wide"
)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    """Load data from Excel file"""
    try:
        if os.path.exists('Dataset_Mahasiswa_Kehadiran_Aktivitas_IPK.xlsx'):
            return pd.read_excel('Dataset_Mahasiswa_Kehadiran_Aktivitas_IPK.xlsx')
        else:
            st.error("File Excel tidak ditemukan!")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==================== APLIKASI UTAMA ====================
def main():
    # Header
    st.title("Prediksi Status Akademik Mahasiswa")
    st.write("Aplikasi sederhana untuk memprediksi status akademik berdasarkan atribut mahasiswa.")
    st.write("---")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar
    menu = st.sidebar.selectbox(
        "Menu",
        ["Data & Statistik", "Prediksi", "Visualisasi"]
    )
    
    st.sidebar.title("Info Data")
    st.sidebar.write("Jumlah data:", len(data))
    st.sidebar.write("Lulus:", int((data['Status Akademik'] == 'Lulus').sum()))
    st.sidebar.write("Tidak Lulus:", int((data['Status Akademik'] == 'Tidak').sum()))
    
    if menu == "Data & Statistik":
        show_data_stats(data)
    elif menu == "Prediksi":
        show_prediction(data)
    elif menu == "Visualisasi":
        show_visualization(data)

# ==================== TAMPILAN DATA & STATISTIK ====================
def show_data_stats(data):
    st.header("Data Mahasiswa")
    st.dataframe(data, use_container_width=True)
    
    st.subheader("Ringkasan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Total mahasiswa")
        st.write(len(data))
    with col2:
        lulus_count = (data['Status Akademik'] == 'Lulus').sum()
        st.write("Lulus")
        st.write(int(lulus_count))
    with col3:
        st.write("Rata-rata IPK")
        st.write(f"{data['IPK'].mean():.2f}")
    
    st.subheader("Statistik Numerik")
    numeric_data = data.select_dtypes(include=[np.number])
    st.dataframe(numeric_data.describe(), use_container_width=True)

# ==================== PREDIKSI ====================
def show_prediction(data):
    st.header("Prediksi Status Akademik")
    
    df = data.copy()
    
    le_sex = LabelEncoder()
    le_marriage = LabelEncoder()
    le_status = LabelEncoder()
    
    df['Jenis Kelamin'] = le_sex.fit_transform(df['Jenis Kelamin'])
    df['Status Menikah'] = le_marriage.fit_transform(df['Status Menikah'])
    df['Status Akademik'] = le_status.fit_transform(df['Status Akademik'])
    
    X = df.drop(['Nama', 'Status Akademik'], axis=1)
    y = df['Status Akademik']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Form input sederhana
    st.subheader("Masukkan data mahasiswa")
    with st.form("prediction_form"):
        nama = st.text_input("Nama")
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        umur = st.number_input("Umur", min_value=17, max_value=100, value=20, step=1)
        status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
        kehadiran = st.slider("Kehadiran (%)", 0, 100, 75)
        partisipasi = st.slider("Partisipasi Diskusi", 0, 100, 70)
        nilai_tugas = st.slider("Nilai Tugas", 0, 100, 75)
        ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
        submitted = st.form_submit_button("Prediksi")
    
    if submitted:
        try:
            input_data = pd.DataFrame({
                'Jenis Kelamin': [le_sex.transform([jenis_kelamin])[0]],
                'Umur': [umur],
                'Status Menikah': [le_marriage.transform([status_menikah])[0]],
                'Kehadiran (%)': [kehadiran],
                'Partisipasi Diskusi (skor)': [partisipasi],
                'Nilai Tugas (rata-rata)': [nilai_tugas],
                'Aktivitas E-Learning (skor)': [75],
                'IPK': [ipk]
            })
            
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            status = le_status.inverse_transform([prediction])[0]
            
            if status == 'Lulus':
                st.success(f"{nama} — diprediksi LULUS")
            else:
                st.error(f"{nama} — diprediksi TIDAK LULUS")
            
            st.write("Probabilitas:")
            st.write(f"Tidak Lulus: {proba[1]*100:.1f}%")
            st.write(f"Lulus: {proba[0]*100:.1f}%")
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.info(f"Akurasi model: {accuracy*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== VISUALISASI ====================
def show_visualization(data):
    st.header("Visualisasi Data")
    chart_type = st.selectbox("Pilih chart", ["Distribusi IPK", "Status Akademik", "Kehadiran vs IPK"])
    
    if chart_type == "Distribusi IPK":
        fig = px.histogram(data, x='IPK', color='Status Akademik', title="Distribusi IPK")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Status Akademik":
        status_counts = data['Status Akademik'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, title="Distribusi Status Akademik")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Kehadiran vs IPK":
        fig = px.scatter(data, x='Kehadiran (%)', y='IPK', color='Status Akademik', title="Kehadiran vs IPK")
        st.plotly_chart(fig, use_container_width=True)

# ==================== RUN APLIKASI ====================
if __name__ == "__main__":
    main()