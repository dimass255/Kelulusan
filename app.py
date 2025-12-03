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

# ==================== CSS SEDERHANA ====================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        padding: 1rem;
    }
    .card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

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
    st.markdown('<h1 class="main-title">🎓 Prediksi Status Akademik Mahasiswa</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar untuk navigasi
    menu = st.sidebar.selectbox(
        "Menu",
        ["📊 Data & Statistik", "🤖 Prediksi", "📈 Visualisasi"]
    )
    
    # Info data di sidebar
    st.sidebar.markdown("### 📋 Info Data")
    st.sidebar.write(f"Jumlah Data: {len(data)}")
    st.sidebar.write(f"Lulus: {(data['Status Akademik'] == 'Lulus').sum()}")
    st.sidebar.write(f"Tidak: {(data['Status Akademik'] == 'Tidak').sum()}")
    
    if menu == "📊 Data & Statistik":
        show_data_stats(data)
    elif menu == "🤖 Prediksi":
        show_prediction(data)
    elif menu == "📈 Visualisasi":
        show_visualization(data)

# ==================== TAMPILAN DATA & STATISTIK ====================
def show_data_stats(data):
    st.markdown("## 📊 Data Mahasiswa")
    
    # Tampilkan data
    st.dataframe(data, use_container_width=True)
    
    # Statistik sederhana
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Mahasiswa", len(data))
    with col2:
        lulus_count = (data['Status Akademik'] == 'Lulus').sum()
        st.metric("Lulus", lulus_count)
    with col3:
        st.metric("Rata IPK", f"{data['IPK'].mean():.2f}")
    
    # Statistik numerik
    st.markdown("### 📈 Statistik Numerik")
    numeric_data = data.select_dtypes(include=[np.number])
    st.dataframe(numeric_data.describe(), use_container_width=True)

# ==================== PREDIKSI ====================
def show_prediction(data):
    st.markdown("## 🤖 Prediksi Status Akademik")
    
    # Preprocess data
    df = data.copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_marriage = LabelEncoder()
    le_status = LabelEncoder()
    
    df['Jenis Kelamin'] = le_sex.fit_transform(df['Jenis Kelamin'])
    df['Status Menikah'] = le_marriage.fit_transform(df['Status Menikah'])
    df['Status Akademik'] = le_status.fit_transform(df['Status Akademik'])
    
    # Prepare data for model
    X = df.drop(['Nama', 'Status Akademik'], axis=1)
    y = df['Status Akademik']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Form input
    st.markdown("### Masukkan Data Mahasiswa")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nama = st.text_input("Nama")
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.slider("Umur", 17, 30, 20)
            status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
        
        with col2:
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 75)
            partisipasi = st.slider("Partisipasi Diskusi", 0, 100, 70)
            nilai_tugas = st.slider("Nilai Tugas", 0, 100, 75)
            ipk = st.slider("IPK", 0.0, 4.0, 3.0, 0.01)
        
        submitted = st.form_submit_button("Prediksi")
    
    if submitted:
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'Jenis Kelamin': [le_sex.transform([jenis_kelamin])[0]],
                'Umur': [umur],
                'Status Menikah': [le_marriage.transform([status_menikah])[0]],
                'Kehadiran (%)': [kehadiran],
                'Partisipasi Diskusi (skor)': [partisipasi],
                'Nilai Tugas (rata-rata)': [nilai_tugas],
                'Aktivitas E-Learning (skor)': [75],  # default value
                'IPK': [ipk]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            
            # Get result
            status = le_status.inverse_transform([prediction])[0]
            
            # Display result
            if status == 'Lulus':
                st.success(f"✅ **{nama}** diprediksi **LULUS**")
            else:
                st.error(f"❌ **{nama}** diprediksi **TIDAK LULUS**")
            
            # Show probabilities
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Peluang Lulus", f"{proba[1]*100:.1f}%")
            with col_prob2:
                st.metric("Peluang Tidak", f"{proba[0]*100:.1f}%")
            
            # Model accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.info(f"**Akurasi Model:** {accuracy*100:.2f}%")
            
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== VISUALISASI ====================
def show_visualization(data):
    st.markdown("## 📈 Visualisasi Data")
    
    # Pilihan chart
    chart_type = st.selectbox(
        "Pilih jenis chart:",
        ["Distribusi IPK", "Status Akademik", "Kehadiran vs IPK"]
    )
    
    if chart_type == "Distribusi IPK":
        fig = px.histogram(
            data, 
            x='IPK', 
            color='Status Akademik',
            title="Distribusi IPK"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Status Akademik":
        status_counts = data['Status Akademik'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Distribusi Status Akademik"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Kehadiran vs IPK":
        fig = px.scatter(
            data,
            x='Kehadiran (%)',
            y='IPK',
            color='Status Akademik',
            title="Hubungan Kehadiran dan IPK"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== RUN APLIKASI ====================
if __name__ == "__main__":
    main()