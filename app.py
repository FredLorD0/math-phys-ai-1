import streamlit as st
import sympy as sp
from sympy.abc import x, t, v, a
from transformers import pipeline

st.set_page_config(page_title="MathPhysAI Bot", layout="centered")
st.title("Math & Physics AI Bot")
st.write("İntegral çöz veya hareket problemi sor, cevaplayayım!")

# NLP modeli yükleniyor (Flan-T5, hızlı ve açık kaynaklı)
@st.cache_resource
def load_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = load_pipeline()

# Kullanıcı seçimi
mode = st.radio("Mod Seç:", ["İntegral Çöz", "Hareket Problemi"])

# ------------------------------
# İntegral Çözüm Modülü
# ------------------------------
if mode == "İntegral Çöz":
    integral_input = st.text_input("İntegral ifadesini gir (örnek: x**2)")
    if st.button("Çöz") and integral_input:
        try:
            expr = sp.sympify(integral_input)
            integral = sp.integrate(expr, x)
            st.latex(f"\\int {sp.latex(expr)}\\,dx = {sp.latex(integral)} + C")
        except Exception as e:
            st.error(f"Hata: {str(e)}")

# ------------------------------
# Hareket Problemi Modülü
# ------------------------------
elif mode == "Hareket Problemi":
    motion_question = st.text_area("Fizik sorusunu gir (örnek: Bir araba 20 m/s hızla 5 saniye giderse kaç metre yol alır?)")
    if st.button("Hesapla") and motion_question:
        try:
            # Sorudan anlam çıkar (gerekli değerler)
            prompt = f"Aşağıdaki fizik sorusunu çözümle ve sonucu yaz: {motion_question}"
            response = qa_pipeline(prompt, max_length=100)[0]['generated_text']
            st.info("Modelin Yorumu:")
            st.write(response)

            # Basit örnekler için otomatik hesap (örnek: s = v * t)
            if "hızla" in motion_question and "saniye" in motion_question:
                import re
                numbers = list(map(float, re.findall(r"[0-9]+\\.?[0-9]*", motion_question)))
                if len(numbers) >= 2:
                    v_val, t_val = numbers[:2]
                    distance = v_val * t_val
                    st.success(f"Yol: {distance} metre")
        except Exception as e:
            st.error(f"Hata: {str(e)}")
