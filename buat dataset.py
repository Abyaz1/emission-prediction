import pandas as pd
from sklearn.model_selection import train_test_split

# Misalnya, dataset disimpan dalam file CSV
data = pd.read_csv('large_dataset_1M_rows.csv')

# ukuran dataset yang diinginkan dalam persen
subset_size = 0.25

# Membagi dataset menjadi subset (20%) dan sisanya (80%)
subset, _ = train_test_split(data, test_size=(1 - subset_size), random_state=42)

# Menampilkan informasi dataset yang lebih kecil
print(f"Jumlah data subset: {len(subset)}")
print(subset.head())

# Menyimpan subset ke file baru jika diperlukan
subset.to_csv('dataset 25%.csv', index=False)
