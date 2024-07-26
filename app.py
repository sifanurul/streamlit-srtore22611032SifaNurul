import pickle
import streamlit as st
import numpy as np

# Memuat model terbaik
with open('model_terbaik.sav', 'rb') as file:
    diabetes_model = pickle.load(file)

# Mengatur halaman dan judul
st.set_page_config(page_title='Prediksi Diabetes', page_icon=':syringe:', layout='centered')
st.markdown("<h1 style='text-align: center; color: #ff6347;'>Prediksi Diabetes dengan Data Mining</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #4682b4;'>Masukkan nilai-nilai berikut untuk memprediksi kemungkinan diabetes:</h4>", unsafe_allow_html=True)

# Input dari pengguna
st.markdown("<h3 style='color: #4682b4;'>Informasi Pasien</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, step=1, help='Jumlah kehamilan yang dialami pasien.')
    Glucose = st.number_input('Glukosa', min_value=0, max_value=200, step=1, help='Kadar glukosa dalam darah.')
    BloodPressure = st.number_input('Tekanan Darah', min_value=0, max_value=150, step=1, help='Tekanan darah pasien.')
    SkinThickness = st.number_input('Ketebalan Kulit', min_value=0, max_value=100, step=1, help='Ketebalan lipatan kulit belakang lengan.')

with col2:
    Insulin = st.number_input('Insulin', min_value=0, max_value=900, step=1, help='Kadar insulin dalam darah.')
    BMI = st.number_input('BMI', min_value=0.000, max_value=70.000, step=0.001, format="%.3f", help='Body Mass Index (Indeks Massa Tubuh).')
    DiabetesPedigreeFunction = st.number_input('Fungsi Diabetes Pedigree', min_value=0.000, max_value=2.500, step=0.001, format="%.3f", help='Riwayat diabetes dalam keluarga.')
    Age = st.number_input('Usia', min_value=0, max_value=120, step=1, help='Usia pasien.')

# Tombol untuk prediksi
st.markdown("<h3 style='color: #4682b4;'>Hasil Prediksi</h3>", unsafe_allow_html=True)
if st.button('Prediksi', key='predict_button'):
    # Mengubah input pengguna menjadi array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=float)

    # Membuat prediksi
    prediction = diabetes_model.predict(input_data)

    # Menampilkan hasil prediksi
    if prediction[0] == 1:
        st.markdown('<div style="padding: 20px; background-color: #ffcccc; border-radius: 10px;"><h3 style="color: red; text-align: center;">Hasil Prediksi: Positif Diabetes</h3></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding: 20px; background-color: #ccffcc; border-radius: 10px;"><h3 style="color: green; text-align: center;">Hasil Prediksi: Negatif Diabetes</h3></div>', unsafe_allow_html=True)

# Menampilkan informasi tambahan
st.markdown("<h3 style='color: #4682b4;'>Informasi Tambahan</h3>", unsafe_allow_html=True)
st.markdown("""
- **Jumlah Kehamilan:** Jumlah kehamilan yang dialami pasien.
- **Glukosa:** Kadar glukosa dalam darah.
- **Tekanan Darah:** Tekanan darah pasien.
- **Ketebalan Kulit:** Ketebalan lipatan kulit belakang lengan.
- **Insulin:** Kadar insulin dalam darah.
- **BMI:** Body Mass Index (Indeks Massa Tubuh).
- **Fungsi Diabetes Pedigree:** Riwayat diabetes dalam keluarga.
- **Usia:** Usia pasien.
""")
