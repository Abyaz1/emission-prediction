import pandas as pd
from sklearn.model_selection import train_test_split

# Membaca dataset (sesuaikan dengan path dataset Anda)
# Misalnya, dataset disimpan dalam file CSV
data = pd.read_csv('large_dataset_1M_rows.csv')

# Menentukan ukuran dataset yang lebih kecil (contoh: 20% dari dataset asli)
subset_size = 0.25

# Membagi dataset menjadi subset (20%) dan sisanya (80%)
subset, _ = train_test_split(data, test_size=(1 - subset_size), random_state=42)

# Menampilkan informasi dataset yang lebih kecil
print(f"Jumlah data subset: {len(subset)}")
print(subset.head())

# Menyimpan subset ke file baru jika diperlukan
subset.to_csv('dataset 250k.csv', index=False)
