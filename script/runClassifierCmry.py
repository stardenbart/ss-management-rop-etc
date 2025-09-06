import re
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CLEANING
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"^(CMI|BMI|PMI|IMI|CML|CM|PM|BM|IM|APD|CGI|FUEL|OPR|RGI)[-\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

# EKSTRAK KATEGORI DARI PREFIX
CATEGORIES = ["CM", "PM", "BM", "IM", "UNK", "APD", "CGI", "FUEL", "OPR", "Other", "CML", "RGI"]

def extract_category(text: str) -> str:
    if not text or not isinstance(text, str):
        return "UNK"
    text = text.upper().strip()

    # Prioritaskan "CML" sebelum "CM"
    if re.match(r"^CML[-\s0-9A-Z]*", text):
        return "CML"
    elif re.match(r"^CM(?!L)[-\s0-9A-Z]*", text):  # CM tapi tidak diikuti L
        return "CM"

    for cat in CATEGORIES:
        if cat in ["CM", "CML"]:
            continue  # sudah di-handle
        if re.match(rf"^{cat}[-\s0-9A-Z]*", text):
            return cat

    return "UNK"

# LOAD MODEL & TOKENIZER & LABEL ENCODERS
model = load_model("best_text_classifier.h5")
with open("tokenizer.pkl", "rb") as f: tokenizer = pickle.load(f)
with open("le_machine.pkl", "rb") as f: le_machine = pickle.load(f)
with open("le_category.pkl", "rb") as f: le_category = pickle.load(f)

max_len = 50

# LOAD SAMPLE DATA
df = pd.read_excel("C:/Users/AbdullahFarauk/OneDrive - PT. Cisarua Mountain Dairy, Tbk/Project/Text Classifier/Training-Nam-Mesin.xlsx")  # ganti sesuai file input
df["clean_text"] = df["DocumentHeaderText"].apply(clean_text)
df["Category_new"] = df["DocumentHeaderText"].apply(extract_category)

# PREPROCESS & PREDICT
X_seq = pad_sequences(tokenizer.texts_to_sequences(df["clean_text"]), maxlen=max_len)

pred_machine, pred_category = model.predict(X_seq)

df["Predicted_Machine"] = le_machine.inverse_transform(np.argmax(pred_machine, axis=1))
df["Predicted_Category"] = le_category.inverse_transform(np.argmax(pred_category, axis=1))

# SAVE HASIL
df.to_excel("prediction_output.xlsx", index=False)
print("✅ Hasil prediksi disimpan ke prediction_output.xlsx")
