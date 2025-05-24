
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt



file_path = 'thailand_co2_emission_1987_2022.csv'
column_year = 'year'
column_emission = 'emissions_tons' 

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



if column_year not in df.columns:
    print(f"Error: Kolom '{column_year}' tidak ditemukan dalam dataset.")
    print(f"Kolom yang tersedia: {df.columns.tolist()}")
    exit()
if column_emission not in df.columns:
    print(f"Error: Kolom '{column_emission}' tidak ditemukan dalam dataset.")
    print(f"Kolom yang tersedia: {df.columns.tolist()}")
    exit()

# Konversi kolom tahun ke datetime 

try:
    df[column_year] = pd.to_datetime(df[column_year], format='%Y')
except ValueError:
    print(f"Error: Format tahun pada kolom '{column_year}' tidak sesuai. Harusnya format YYYY (misal: 1990).")
    
    try:
        df[column_year] = df[column_year].astype(int)
        df[column_year] = pd.to_datetime(df[column_year], format='%Y')
        print(f"Kolom '{column_year}' berhasil dikonversi ke datetime setelah diperlakukan sebagai integer.")
    except Exception as e_inner:
        print(f"Gagal mengkonversi kolom '{column_year}' ke datetime: {e_inner}")
        exit()

df.set_index(column_year, inplace=True)
df.sort_index(inplace=True) 
data = df[[column_emission]].copy()
data.rename(columns={column_emission: 'CO2_Emission'}, inplace=True)
data['Lag_1'] = data['CO2_Emission'].shift(1)
data.dropna(inplace=True)
print(data.head())

#  Pembagian Data (Training dan Testing Set)

X = data[['Lag_1']]
y = data['CO2_Emission']

split_ratio = 0.8
split_index = int(len(data) * split_ratio)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\nUkuran data training: {X_train.shape[0]} baris")
print(f"Ukuran data testing: {X_test.shape[0]} baris")

# Pelatihan Model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Koefisien (slope): {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Membuat Prediksi
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluasi Model
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

# Visualisasi Hasil
plt.figure(figsize=(14, 7))

# Plot data training, data testing, dan prediksi
plt.plot(y_train.index, y_train, label='Data Training Aktual', color='blue')
plt.plot(y_test.index, y_test, label='Data Testing Aktual', color='green')
plt.plot(y_test.index, y_pred_test, label='Prediksi Model pada Data Testing', color='red', linestyle='--')
# plt.plot(y_train.index, y_pred_train, label='Prediksi Model pada Data Training', color='orange', linestyle=':') # Opsional: plot prediksi training

plt.title('Prediksi Emisi CO2 Thailand vs Data Aktual')
plt.xlabel('Tahun')
plt.ylabel('Emisi CO2 (Total)'
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
