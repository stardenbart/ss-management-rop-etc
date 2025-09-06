import pandas as pd
import re
import random
import nltk
from sklearn.model_selection import train_test_split

# pastiin download wordnet dulu
nltk.download('wordnet')
from nltk.corpus import wordnet

# --- fungsi clean text ---
def clean_text(text: str) -> str:
    text = str(text)
    # hapus prefix kategori (CM, PM, IM, BM, dll)
    text = re.sub(r"^(CMI|BMI|PMI|IMI|CML|CM|PM|BM|IM)[-\s]*", "", text, flags=re.IGNORECASE)
    # keep huruf & angka
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    # rapikan spasi
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.replace(" ", "")   # versi no-spasi total

# --- fungsi text augmentation ---
def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name().replace("_", "")
            new_words = [synonym if w == random_word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)

def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 1:
        return text
    new_words = [w for w in words if random.uniform(0,1) > p]
    if len(new_words) == 0:
        return random.choice(words)
    return " ".join(new_words)

def random_swap(text, n=1):
    words = text.split()
    if len(words) < 2:   # kalo kata kurang dari 2, return langsung
        return text
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return " ".join(new_words)

def augment_text(text, num_aug=3):
    aug_texts = []
    if len(text.split()) < 2:
        return [text]  # biarin aja kalo terlalu pendek

    for _ in range(num_aug):
        choice = random.choice(["synonym", "swap", "delete"])
        if choice == "synonym":
            aug_texts.append(synonym_replacement(text))
        elif choice == "swap":
            aug_texts.append(random_swap(text))
        else:
            aug_texts.append(random_deletion(text))
    return aug_texts

# --- load dataset ---
file_path = "C:/Users/AbdullahFarauk/OneDrive - PT. Cisarua Mountain Dairy, Tbk/Project/Text Classifier/train_data_v9.csv"  # ganti sesuai file
df = pd.read_csv(file_path)

# pastiin kolomnya sesuai
df["clean_text"] = df["DocumentHeaderText"].apply(clean_text)

# augmentasi data
augmented_rows = []
for _, row in df.iterrows():
    text = row["clean_text"]
    label = row["Machine"]
    original_text = row["DocumentHeaderText"]  # simpan text asli
    for aug in augment_text(text, num_aug=2):  # tiap teks jadi 2 tambahan
        augmented_rows.append({
            "DocumentHeaderText": original_text,  # isi lagi biar nggak NaN
            "clean_text": aug,
            "Machine": label
        })

df_aug = pd.DataFrame(augmented_rows)

# gabungkan data asli + augmentasi
df_final = pd.concat([df, df_aug], ignore_index=True)

print(f"🔍 Data asli: {len(df)}")
print(f"✨ Data augmentasi: {len(df_aug)}")
print(f"📊 Total data: {len(df_final)}")

# split train-test
train_df, test_df = train_test_split(df_final, test_size=0.2, stratify=df_final["Machine"], random_state=42)

print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")

# save hasil siap training
train_df.to_csv("train_data_v10.csv", index=False)
test_df.to_csv("test_data_v10.csv", index=False)