import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Simulasi dataset awal
np.random.seed(42)
jumlah_data = 10000
jenis_pupuk = ["Organik", "Kimia", "Campuran"]
pola_irigasi = ["Tetes", "Curah", "Furrow"]
produktivitas = ["Rendah", "Sedang", "Tinggi"]

data = {
    "Jenis Pupuk": np.random.choice(jenis_pupuk, jumlah_data, p=[0.4, 0.3, 0.3]),
    "Pola Irigasi": np.random.choice(pola_irigasi, jumlah_data, p=[0.3, 0.4, 0.3]),
    "Curah Hujan (mm)": np.random.randint(500, 2000, jumlah_data),
    "PH Tanah": np.random.uniform(4.5, 8.5, jumlah_data),
    "Produktivitas": np.random.choice(produktivitas, jumlah_data, p=[0.3, 0.4, 0.3])
}

df = pd.DataFrame(data)

# Fungsi visualisasi data
def visualize_data(df):
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    sns.countplot(x="Jenis Pupuk", data=df, palette="viridis")
    plt.title("Distribusi Jenis Pupuk")
    plt.xlabel("Jenis Pupuk")
    plt.ylabel("Jumlah")

    plt.subplot(1, 3, 2)
    sns.countplot(x="Pola Irigasi", data=df, palette="viridis")
    plt.title("Distribusi Pola Irigasi")
    plt.xlabel("Pola Irigasi")
    plt.ylabel("Jumlah")

    plt.subplot(1, 3, 3)
    sns.countplot(x="Produktivitas", data=df, palette="viridis")
    plt.title("Distribusi Produktivitas")
    plt.xlabel("Produktivitas")
    plt.ylabel("Jumlah")
    
    plt.tight_layout()
    plt.show()

# Fungsi untuk melatih model
def train_model(df):
    # Encoding variabel kategori
    df_encoded = pd.get_dummies(df, columns=["Jenis Pupuk", "Pola Irigasi"], drop_first=True)
    df_encoded["Produktivitas"] = df["Produktivitas"].map({"Rendah": 0, "Sedang": 1, "Tinggi": 2})
    
    # Pisahkan fitur dan label
    X = df_encoded.drop(columns=["Produktivitas"])
    y = df_encoded["Produktivitas"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=20, min_samples_split=5)
    model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # Simpan model
    joblib.dump(model, 'model_produktivitas.pkl')
    print("Model berhasil dilatih dan disimpan.")
    
    return model

# Fungsi untuk menambahkan data baru
def add_new_data(df):
    print("\n=== Tambah Data Baru ===")
    jenis_pupuk = input("Masukkan Jenis Pupuk (Organik, Kimia, Campuran): ").strip().capitalize()
    while jenis_pupuk not in jenis_pupuk_list:
        print("Input tidak valid. Silakan masukkan salah satu dari: Organik, Kimia, Campuran.")
        jenis_pupuk = input("Masukkan Jenis Pupuk (Organik, Kimia, Campuran): ").strip().capitalize()
    
    pola_irigasi = input("Masukkan Pola Irigasi (Tetes, Curah, Furrow): ").strip().capitalize()
    while pola_irigasi not in pola_irigasi_list:
        print("Input tidak valid. Silakan masukkan salah satu dari: Tetes, Curah, Furrow.")
        pola_irigasi = input("Masukkan Pola Irigasi (Tetes, Curah, Furrow): ").strip().capitalize()
    
    try:
        curah_hujan = float(input("Masukkan Curah Hujan (mm): "))
    except ValueError:
        print("Input Curah Hujan harus berupa angka. Data tidak ditambahkan.")
        return df
    
    try:
        ph_tanah = float(input("Masukkan pH Tanah: "))
    except ValueError:
        print("Input pH Tanah harus berupa angka. Data tidak ditambahkan.")
        return df
    
    produktivitas_input = input("Masukkan Produktivitas (Rendah, Sedang, Tinggi): ").strip().capitalize()
    while produktivitas_input not in produktivitas_list:
        print("Input tidak valid. Silakan masukkan salah satu dari: Rendah, Sedang, Tinggi.")
        produktivitas_input = input("Masukkan Produktivitas (Rendah, Sedang, Tinggi): ").strip().capitalize()
    
    # Menambahkan data baru ke DataFrame
    new_data = {
        "Jenis Pupuk": jenis_pupuk,
        "Pola Irigasi": pola_irigasi,
        "Curah Hujan (mm)": curah_hujan,
        "PH Tanah": ph_tanah,
        "Produktivitas": produktivitas_input
    }
    df_new = pd.DataFrame([new_data])
    
    # Gunakan pd.concat untuk menambahkan data
    df = pd.concat([df, df_new], ignore_index=True)
    print("\nData baru berhasil ditambahkan.")
    return df

# Fungsi prediksi
def predict_productivity(df_encoded_columns):
    # Load model
    try:
        model = joblib.load('model_produktivitas.pkl')
    except FileNotFoundError:
        print("Model belum dilatih. Silakan latih model terlebih dahulu.")
        return
    
    # Input pengguna
    print("\n=== Prediksi Produktivitas Sawah ===")
    jenis_pupuk = input("Masukkan Jenis Pupuk (Organik, Kimia, Campuran): ").strip().capitalize()
    while jenis_pupuk not in jenis_pupuk_list:
        print("Input tidak valid. Silakan masukkan salah satu dari: Organik, Kimia, Campuran.")
        jenis_pupuk = input("Masukkan Jenis Pupuk (Organik, Kimia, Campuran): ").strip().capitalize()
    
    pola_irigasi = input("Masukkan Pola Irigasi (Tetes, Curah, Furrow): ").strip().capitalize()
    while pola_irigasi not in pola_irigasi_list:
        print("Input tidak valid. Silakan masukkan salah satu dari: Tetes, Curah, Furrow.")
        pola_irigasi = input("Masukkan Pola Irigasi (Tetes, Curah, Furrow): ").strip().capitalize()
    
    try:
        curah_hujan = float(input("Masukkan Curah Hujan (mm): "))
    except ValueError:
        print("Input Curah Hujan harus berupa angka.")
        return
    
    try:
        ph_tanah = float(input("Masukkan pH Tanah: "))
    except ValueError:
        print("Input pH Tanah harus berupa angka.")
        return
    
    # Buat DataFrame input dengan encoding yang sesuai
    input_data = {
        "Curah Hujan (mm)": [curah_hujan],
        "PH Tanah": [ph_tanah],
    }
    
    # Encode variabel kategori dengan menggunakan get_dummies dan align dengan kolom model
    jenis_pupuk_dummies = pd.get_dummies([jenis_pupuk], prefix="Jenis Pupuk", drop_first=True)
    pola_irigasi_dummies = pd.get_dummies([pola_irigasi], prefix="Pola Irigasi", drop_first=True)
    
    input_encoded = pd.concat([pd.DataFrame(input_data), jenis_pupuk_dummies, pola_irigasi_dummies], axis=1)
    
    # Pastikan semua kolom yang diperlukan ada
    for col in df_encoded_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Urutkan kolom sesuai dengan model
    input_encoded = input_encoded[df_encoded_columns]
    
    # Prediksi
    prediction = model.predict(input_encoded)
    mapping = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
    result = mapping.get(prediction[0], "Tidak Diketahui")
    print(f"\nHasil Prediksi Produktivitas: {result}")

# Inisialisasi daftar validasi
jenis_pupuk_list = ["Organik", "Kimia", "Campuran"]
pola_irigasi_list = ["Tetes", "Curah", "Furrow"]
produktivitas_list = ["Rendah", "Sedang", "Tinggi"]

# Fungsi utama
def main():
    global df  # agar perubahan pada df dapat diakses di seluruh fungsi
    
    while True:
        print("\n=== Menu ===")
        print("1. Lihat Visualisasi Data")
        print("2. Tambah Data Baru")
        print("3. Latih Ulang Model")
        print("4. Prediksi Produktivitas")
        print("5. Keluar")
        choice = input("Pilih menu: ")
        
        if choice == "1":
            visualize_data(df)
        elif choice == "2":
            df = add_new_data(df)
        elif choice == "3":
            train_model(df)
        elif choice == "4":
            # Pastikan model sudah dilatih dan simpan kolom encoding
            try:
                model = joblib.load('model_produktivitas.pkl')
                # Membuat encoding untuk mendapatkan kolom yang sama dengan model
                df_encoded = pd.get_dummies(df, columns=["Jenis Pupuk", "Pola Irigasi"], drop_first=True)
                df_encoded["Produktivitas"] = df["Produktivitas"].map({"Rendah": 0, "Sedang": 1, "Tinggi": 2})
                # Simpan kolom yang digunakan oleh model
                df_encoded_columns = df_encoded.drop(columns=["Produktivitas"]).columns
                predict_productivity(df_encoded_columns)
            except FileNotFoundError:
                print("Model belum dilatih. Silakan latih model terlebih dahulu.")
        elif choice == "5":
            print("Program selesai.")
            break
        else:
            print("Pilihan tidak valid. Coba lagi.")

# Jalankan program utama
if __name__ == "__main__":
    main()
