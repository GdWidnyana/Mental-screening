import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Memuat model dan encoder
model = tf.keras.models.load_model("mental_disorder_model.keras")

# Memuat scaler dan label encoder
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Mapping label ke Bahasa Indonesia
label_mapping = {
    'MDD': 'Gangguan Depresi Mayor (MDD)',  # Mapping ke Bahasa Indonesia
    'ASD': 'Gangguan Spektrum Autisme (ASD)',
    'Loneliness': 'Kesepian',
    'bipolar': 'Gangguan Bipolar',
    'anexiety': 'Gangguan Kecemasan',
    'PTSD': 'Gangguan Stres Pascatrauma (PTSD)',
    'sleeping disorder': 'Gangguan Tidur',
    'psychotic deprission': 'Depresi Psikotik',
    'eating disorder': 'Gangguan Makan',
    'ADHD': 'Gangguan Pemusatan Perhatian dan Hiperaktivitas (ADHD)',
    'PDD': 'Gangguan Perkembangan Pervasif (PDD)',
    'OCD': 'Gangguan Obsesif-Kompulsif (OCD)',
}

# Aplikasi Streamlit
st.title("Prediksi Gangguan Mental")

# Input pengguna
ag = st.text_input("Silakan masukkan usia Anda:", "")  # Biarkan kosong
ag = int(ag) if ag.isdigit() else None  # Validasi jika usia adalah angka, atau None jika kosong

# Pertanyaan dengan jawaban Ya atau Tidak menggunakan radio button dengan key unik
feeling_nervous = st.radio("Apakah Anda merasa cemas?", ("Tidak", "Ya"), key="feeling_nervous")
panic = st.radio("Apakah Anda mengalami panik?", ("Tidak", "Ya"), key="panic")
breathing_rapidly = st.radio("Apakah Anda bernapas cepat?", ("Tidak", "Ya"), key="breathing_rapidly")
sweating = st.radio("Apakah Anda berkeringat?", ("Tidak", "Ya"), key="sweating")
trouble_in_concentration = st.radio("Apakah Anda mengalami kesulitan berkonsentrasi?", ("Tidak", "Ya"), key="trouble_in_concentration")
having_trouble_in_sleeping = st.radio("Apakah Anda mengalami kesulitan tidur?", ("Tidak", "Ya"), key="having_trouble_in_sleeping")
having_trouble_with_work = st.radio("Apakah Anda mengalami kesulitan dalam bekerja?", ("Tidak", "Ya"), key="having_trouble_with_work")
hopelessness = st.radio("Apakah Anda merasa putus asa?", ("Tidak", "Ya"), key="hopelessness")
anger = st.radio("Apakah Anda sering merasa marah?", ("Tidak", "Ya"), key="anger")
over_react = st.radio("Apakah Anda mudah bereaksi berlebihan?", ("Tidak", "Ya"), key="over_react")
change_in_eating = st.radio("Apakah ada perubahan pola makan?", ("Tidak", "Ya"), key="change_in_eating")
suicidal_thought = st.radio("Apakah Anda pernah memiliki pikiran untuk bunuh diri?", ("Tidak", "Ya"), key="suicidal_thought")
feeling_tired = st.radio("Apakah Anda sering merasa lelah?", ("Tidak", "Ya"), key="feeling_tired")
close_friend = st.radio("Apakah Anda memiliki teman dekat?", ("Tidak", "Ya"), key="close_friend")
social_media_addiction = st.radio("Apakah Anda kecanduan media sosial?", ("Tidak", "Ya"), key="social_media_addiction")
weight_gain = st.radio("Apakah Anda mengalami kenaikan berat badan?", ("Tidak", "Ya"), key="weight_gain")
introvert = st.radio("Apakah Anda seorang introvert?", ("Tidak", "Ya"), key="introvert")
popping_up_stressful_memory = st.radio("Apakah sering muncul ingatan yang membuat stres?", ("Tidak", "Ya"), key="popping_up_stressful_memory")
having_nightmares = st.radio("Apakah Anda sering mengalami mimpi buruk?", ("Tidak", "Ya"), key="having_nightmares")
avoids_people_or_activities = st.radio("Apakah Anda menghindari orang atau aktivitas tertentu?", ("Tidak", "Ya"), key="avoids_people_or_activities")
feeling_negative = st.radio("Apakah Anda sering merasa negatif?", ("Tidak", "Ya"), key="feeling_negative")
trouble_concentrating = st.radio("Apakah Anda mengalami kesulitan dalam fokus berinteraksi?", ("Tidak", "Ya"), key="trouble_concentrating")
blaming_yourself = st.radio("Apakah Anda sering menyalahkan diri sendiri?", ("Tidak", "Ya"), key="blaming_yourself")
hallucinations = st.radio("Apakah Anda mengalami halusinasi?", ("Tidak", "Ya"), key="hallucinations")
repetitive_behaviour = st.radio("Apakah Anda memiliki perilaku yang repetitif?", ("Tidak", "Ya"), key="repetitive_behaviour")
seasonally = st.radio("Apakah Anda merasa berbeda tergantung musim?", ("Tidak", "Ya"), key="seasonally")
increased_energy = st.radio("Apakah Anda merasa memiliki energi yang berlebihan?", ("Tidak", "Ya"), key="increased_energy")

# Tombol submit untuk memproses data setelah input
if st.button("Submit"):
    # Cek jika usia telah diisi dengan benar
    if ag is None:
        st.write("Harap masukkan usia Anda dengan benar.")
    else:
        # Konversi input pengguna menjadi array numpy
        user_data = np.array([[ag, 
                               feeling_nervous == "Ya", panic == "Ya", breathing_rapidly == "Ya", sweating == "Ya",
                               trouble_in_concentration == "Ya", having_trouble_in_sleeping == "Ya", having_trouble_with_work == "Ya",
                               hopelessness == "Ya", anger == "Ya", over_react == "Ya", change_in_eating == "Ya", suicidal_thought == "Ya",
                               feeling_tired == "Ya", close_friend == "Ya", social_media_addiction == "Ya", weight_gain == "Ya", introvert == "Ya",
                               popping_up_stressful_memory == "Ya", having_nightmares == "Ya", avoids_people_or_activities == "Ya",
                               feeling_negative == "Ya", trouble_concentrating == "Ya", blaming_yourself == "Ya", hallucinations == "Ya",
                               repetitive_behaviour == "Ya", seasonally == "Ya", increased_energy == "Ya"]]).astype(float)

        # Melakukan skalasi pada data pengguna
        user_data_scaled = scaler.transform(user_data)

        # Prediksi
        prediction = model.predict(user_data_scaled)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

        # Map ke Bahasa Indonesia
        predicted_class_in_indonesian = label_mapping.get(predicted_class[0], predicted_class[0])

        # Tampilkan hasil prediksi
        st.write(f"Hasil Diagnosa bahwa Anda mengalami {predicted_class_in_indonesian}")
