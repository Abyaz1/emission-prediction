# Impor library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# --- 1. Pemuatan Data ---
# Ganti 'nama_file.csv' dengan nama file dataset Anda
# Ganti 'Nama_Kolom_Tahun' dan 'Nama_Kolom_Emisi' sesuai dengan dataset Anda
file_path = 'thailand_co2_emission_1987_2022.csv'
column_year = 'year'  # Ganti jika nama kolom tahun berbeda
column_emission = 'emissions_tons' # Ganti jika nama kolom emisi CO2 berbeda (misal: 'CO2 Emission', 'Emisi')

try:
    df = pd.read_csv(file_path)
    print("Data berhasil dimuat.")
    print("5 baris pertama data:")
    print(df.head())
    print("\nInfo dataset:")
    df.info()
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan. Pastikan path file sudah benar.")
    exit()
except Exception as e:
    print(f"Terjadi error saat memuat data: {e}")
    exit()

# --- 2. Pra-pemrosesan Dasar ---
# Pastikan kolom tahun dan emisi ada
if column_year not in df.columns:
    print(f"Error: Kolom '{column_year}' tidak ditemukan dalam dataset.")
    print(f"Kolom yang tersedia: {df.columns.tolist()}")
    exit()
if column_emission not in df.columns:
    print(f"Error: Kolom '{column_emission}' tidak ditemukan dalam dataset.")
    print(f"Kolom yang tersedia: {df.columns.tolist()}")
    exit()

# Konversi kolom tahun ke datetime (jika belum) dan jadikan index
# Asumsi kolom tahun hanya berisi tahun (misal: 1987, 1988)
try:
    df[column_year] = pd.to_datetime(df[column_year], format='%Y')
except ValueError:
    print(f"Error: Format tahun pada kolom '{column_year}' tidak sesuai. Harusnya format YYYY (misal: 1990).")
    # Coba konversi sebagai integer jika format='%Y' gagal dan kolomnya numerik
    try:
        df[column_year] = df[column_year].astype(int)
        df[column_year] = pd.to_datetime(df[column_year], format='%Y')
        print(f"Kolom '{column_year}' berhasil dikonversi ke datetime setelah diperlakukan sebagai integer.")
    except Exception as e_inner:
        print(f"Gagal mengkonversi kolom '{column_year}' ke datetime: {e_inner}")
        exit()

df.set_index(column_year, inplace=True)
df.sort_index(inplace=True) # Pastikan data terurut berdasarkan waktu

# Pilih hanya kolom emisi untuk analisis (jika ada kolom lain)
data = df[[column_emission]].copy()
data.rename(columns={column_emission: 'CO2_Emission'}, inplace=True) # Standarisasi nama kolom target

# Cek missing values di kolom target awal
if data['CO2_Emission'].isnull().any():
    print(f"\nWarning: Ada nilai hilang pada kolom target '{column_emission}'.")
    print("Jumlah nilai hilang:", data['CO2_Emission'].isnull().sum())
    # Pertimbangkan strategi penanganan missing value (misal: imputasi) sebelum membuat fitur lag
    # Untuk contoh ini, kita akan lanjutkan, tapi ini bisa mempengaruhi hasil.
    # data['CO2_Emission'].fillna(method='ffill', inplace=True) # Contoh: forward fill

# --- 3. Feature Engineering (Membuat Fitur Lag) ---
# Kita akan membuat fitur lag 1 (emisi tahun sebelumnya)
# Anda bisa menambahkan lebih banyak lag (misal, lag_2, lag_3)
data['Lag_1'] = data['CO2_Emission'].shift(1)
# data['Lag_2'] = data['CO2_Emission'].shift(2) # Contoh jika ingin menambah lag_2

# Hapus baris yang memiliki NaN (akibat dari shift/lag)
data.dropna(inplace=True)

if data.empty:
    print("Error: Tidak ada data tersisa setelah membuat fitur lag dan menghapus NaN.")
    print("Ini bisa terjadi jika dataset terlalu kecil atau semua nilai CO2_Emission adalah NaN.")
    exit()

print("\nData setelah feature engineering (dengan fitur lag):")
print(data.head())

# --- 4. Pembagian Data (Training dan Testing Set) ---
# Data akan dibagi secara kronologis
# X adalah fitur (Lag_1), y adalah target (CO2_Emission)
X = data[['Lag_1']] # Jika ada lebih banyak fitur lag, tambahkan di sini: [['Lag_1', 'Lag_2']]
y = data['CO2_Emission']

# Tentukan titik pembagian (misalnya, 80% untuk training, 20% untuk testing)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

if X_train.empty or y_train.empty:
    print("Error: Data training kosong. Periksa ukuran dataset dan rasio pembagian.")
    exit()
if X_test.empty or y_test.empty:
    print("Error: Data testing kosong. Periksa ukuran dataset dan rasio pembagian.")
    exit()

print(f"\nUkuran data training: {X_train.shape[0]} baris")
print(f"Ukuran data testing: {X_test.shape[0]} baris")

# --- 5. Pelatihan Model Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel berhasil dilatih.")
print(f"Koefisien (slope): {model.coef_}")
print(f"Intercept: {model.intercept_}")

# --- 6. Membuat Prediksi ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- 7. Evaluasi Model ---
# Evaluasi pada data training
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
print(f"\nEvaluasi pada Data Training:")
print(f"RMSE Training: {rmse_train:.2f}")
print(f"MAE Training: {mae_train:.2f}")

# Evaluasi pada data testing
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
print(f"\nEvaluasi pada Data Testing:")
print(f"RMSE Testing: {rmse_test:.2f}")
print(f"MAE Testing: {mae_test:.2f}")

# --- 8. Visualisasi Hasil ---
plt.figure(figsize=(14, 7))

# Plot data training, data testing, dan prediksi
plt.plot(y_train.index, y_train, label='Data Training Aktual', color='blue')
plt.plot(y_test.index, y_test, label='Data Testing Aktual', color='green')
plt.plot(y_test.index, y_pred_test, label='Prediksi Model pada Data Testing', color='red', linestyle='--')
# plt.plot(y_train.index, y_pred_train, label='Prediksi Model pada Data Training', color='orange', linestyle=':') # Opsional: plot prediksi training

plt.title('Prediksi Emisi CO2 Thailand vs Data Aktual')
plt.xlabel('Tahun')
plt.ylabel('Emisi CO2 (Total)') # Sesuaikan unit jika diketahui
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Membuat DataFrame untuk perbandingan aktual vs prediksi pada data test
results_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred_test})
results_df['Error'] = results_df['Aktual'] - results_df['Prediksi']
results_df['Absolute_Error'] = np.abs(results_df['Error'])
print("\nPerbandingan Aktual vs Prediksi pada Data Testing:")
print(results_df)

print("\nAnalisis selesai.")
