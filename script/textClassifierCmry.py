import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# CLEANING
def clean_text(text: str) -> str:
    text = str(text)
    # Hapus karakter non huruf/angka kecuali spasi
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

# EKSTRAK KATEGORI DARI PREFIX
CATEGORIES = ["CM", "PM", "BM", "IM", "APD", "CGI", "FUEL", "OPR", "CML", "RGI"]

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

# LOAD DATA
df = pd.read_csv(
    "C:/Users/AbdullahFarauk/OneDrive - PT. Cisarua Mountain Dairy, Tbk/Project/Text Classifier/training_v11.csv"
)

df["clean_text"] = df["DocumentHeaderText"].apply(clean_text)
df["Category_new"] = df["DocumentHeaderText"].apply(extract_category)

print(df[["DocumentHeaderText","clean_text","Machine","Category_new"]].head())

# LABEL ENCODING
le_machine = LabelEncoder()
df["Machine_enc"] = le_machine.fit_transform(df["Machine"])

le_cat = LabelEncoder()
df["Category_enc"] = le_cat.fit_transform(df["Category_new"])

# SPLIT DATA
X_train, X_test, y_train_m, y_test_m, y_train_c, y_test_c = train_test_split(
    df["clean_text"], df["Machine_enc"], df["Category_enc"], 
    test_size=0.2, random_state=42
)

# TOKENIZER
max_words = 5000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

y_train_m = np.array(y_train_m)
y_test_m = np.array(y_test_m)
y_train_c = np.array(y_train_c)
y_test_c = np.array(y_test_c)

# MODEL MULTI-OUTPUT
input_layer = Input(shape=(max_len,))
embedding = Embedding(max_words, 128)(input_layer)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedding)

# Output 1: Machine
out_machine = Dense(len(le_machine.classes_), activation="softmax", name="machine_output")(x)

# Output 2: Category
out_category = Dense(len(le_cat.classes_), activation="softmax", name="category_output")(x)

model = Model(inputs=input_layer, outputs=[out_machine, out_category])
model.compile(
    loss={
        "machine_output": "sparse_categorical_crossentropy",
        "category_output": "sparse_categorical_crossentropy",
    },
    optimizer="adam",
    metrics={
        "machine_output": "accuracy",
        "category_output": "accuracy",
    }
)

model.summary()

# TRAINING + CALLBACKS
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("best_text_classifier.h5", monitor="val_loss", save_best_only=True)
]

history = model.fit(
    X_train_seq, {"machine_output": y_train_m, "category_output": y_train_c},
    validation_data=(X_test_seq, {"machine_output": y_test_m, "category_output": y_test_c}),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# SAVE FINAL MODEL + ENCODERS
model.save("final_text_classifier.h5")

with open("tokenizer.pkl","wb") as f: pickle.dump(tokenizer, f)
with open("le_machine.pkl","wb") as f: pickle.dump(le_machine, f)
with open("le_category.pkl","wb") as f: pickle.dump(le_cat, f)

print("✅ Training selesai. Model & encoder sudah disimpan.")
