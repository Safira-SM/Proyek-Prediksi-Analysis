# Laporan Proyek Machine Learning - Safira Maulidia

## Domain Proyek

Pada bidang property ada banyak pilihan yang menjadi alasan untuk konsumen membeli product, seperti luas bangunan, jumlah kamar mandi, jumlah kamar tidur dan lain sebagainya. Dan dari alasan-alasan ini akan menjadi faktor yang mempengaruhi harga dan keputusan konsumen. Dan saya melalui peneran machine learning yang sudah diajarkan dan saya ketahui ingin memberikan insight terkait faktor-faktor tersebut. 


**Rubrik/Kriteria Tambahan (Opsional)**:
Mengapa masalah ini penting, karena harga product property itu sangat variatif dan sulit untuk diprediksi. Maka dari itu saya membangun model prediksi harga berbasis data histori agar dapat mengidentifikasi fitur- fitur yang paling mempengaruhi keputusan.


## Business Understanding


### Problem Statements

- Fitur apa saja yang paling memengaruhhi keputusan konsumen untuk membeli property ?
- Apakah property dengan karakteristik tertentu bisa menyebabkan harga yang secara signifikan lebih tinggi ?
- Model regresi mana yang lebih baik dalam memprediksi harga property ?


### Goals

- Mengidentifikasi fitur property yang paling memengaruhi harga
- Menguji dengan melakukan hipotesis berdasarkan segmentasi fitur
- Membandingkan kinerja dua model 



**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution statements
- Menggunakan uji hipotesis untuk menganalisis pengaruh fitur terhadap harga
- Membangun model regresi linier dan decision tree untuk memprediksi harga
- Mengembangan odel random forest untuk meningkatkan preforma dan mengevaluasi importance score 

## Data Understanding
Dataset yang digunakan adalah Melbourne Housing Market. <br>
Link dataset : https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot 

Informasi Umum Dataset
- Jumlah baris: 13.580
- Jumlah kolom: 21

Kondisi Dataset Awal :
1. Dataset memiliki missing value di beberapa kolom :
- Car
- BuildingArea
- YearBuilt
- CouncilArea

2. Melakukan pengecekan missing value dengan `df.isnull().sum()`
3. Tidak terdapat data duplikat yang signifikan

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Suburb: Nama daerah/lokasi properti berada.
- Address: Alamat lengkap properti.
- Rooms: Jumlah total ruangan 
- Type: Jenis properti 
- Price: Harga properti 
- Method: Metode penjualan 
- SellerG: Nama agen atau perusahaan penjual.
- Date: Tanggal penjualan.
- Distance: Jarak properti ke pusat kota.
- Postcode: Kode pos area properti.
- Bedroom2: Jumlah kamar tidur.
- Bathroom: Jumlah kamar mandi.
- Car: Kapasitas parkir/garasi.
- Landsize: Luas tanah.
- BuildingArea: Luas bangunan.
- YearBuilt: Tahun bangunan dibangun.
- CouncilArea: Wilayah administratif properti.
- Lattitude: Koordinat lintang properti.
- Longtitude: Koordinat bujur properti.
- Regionname: Nama wilayah besar.
- Propertycount: Jumlah properti dalam area tertentu.
 

**Rubrik/Kriteria Tambahan (Opsional)**:
### EDA (exploratory data analysis).
- Distribusi harga yang sangat miring 
- Terdapat outlier yang cukup parah pada fitur landsize dan building area
- Korelasi tertinggi terhadap harga adalah building area dan landsize


## Data Preparation
Langkah teknik yang dilakukan, yaitu 

**Rubrik/Kriteria Tambahan (Opsional)**: 
1. Menghapus Data Duplikat
`df.drop_duplicates(inplace=True)`

2. Menghapus kolom CoouncilArea
`df.dropna(subset=['CouncilArea'], inplace=True)`

3. Menghapus baris missing value pada kolom Building dan Car
`df = df[df['BuildingArea'].notnull() & df['Car'].notnull()]`

4. Mengisi nilai kosong dengan median pada kolom YearBuilt
`median_year = df['YearBuilt'].median()
df.loc[:,'YearBuilt'] = df['YearBuilt'].fillna(median_year)`

5. Pembagian dataset (ini saya lakukan di bagian pemodelan)
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
Proporsi 80:20 yang digunakan untuk data latih dan data uji. 

## Modeling
Model yang digunakan : 
- Linear Regression 
- Decision Tree Reressor 
- Random Forest Regressor 

**Rubrik/Kriteria Tambahan (Opsional)**: 
1. Linear Regression 
- Cara kerjanya : Mencari garis regresi terbaik yang meminimalkan selisih kuadrat (least squares) antara nilai aktual dan nilai prediksi.
- Kelebihan nya : interpretas mudah, baseline model
- Kekurangannya : tidak menangkap non linearitas  
parameter yang saya gunakan default : `fit_intercept=True, copy_X=True, n_jobs=None, positive=False`


2. Decision Tree Reressor 
- Cara kerjanya : Membagi data ke dalam subset berdasar fitur yang memberikan pengurangan error paling besar, hingga mencapai kondisi akhir atau batas tertentu.
- Kelebihan : Menangkap non linearitas, feature importance 
- Kekurangan : rentang overfitting  
parameter yang saya gunakan default : `criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None`, dan lainnya.


3. Random Forest Regressor 
- Cara kerjanya : Menggabungkan banyak decision tree (ensemble) dan menghasilkan prediksi rata-rata dari semua tree untuk meningkatkan akurasi dan mengurangi overfitting.
- Kelebihan : lebih stabil, akurasi tinggi, menangani outlier
- Kekurangan : interpretasi lebih kompleks 

Model terbaik itu ada di Random Forest, dengan parameter `n_estimonitors=100, max_depth=10, random_state=42`


## Evaluation
Metrik Evaluasi :
- RMSE : mengukur selisih antara prediksi dan nilai aktual 
- R2 Score : Menjelaskan proporsi variasi terget yang dapat di jelaskan oleh model 

(Model, RMSE, R2 Score)
1. Linear Regression 
- RMSE: 550781.2410914198
- R2 Score: 0.42985999399735697

2. Decision Tree
- RMSE: 625150.490475077
- R2 Score: 0.26549900605014154

3. Random Forest
- RMSE: 490011.35672286485
- R2 Score: 0.5487309731763566

Feature Importance :
1. Building Area
2. YearBuilt
3. Landsize
4. Roms
5. Bathroom
6. Car

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Dari problem statement terjawab bahwa model berhasil mengidentifikasi fitur penting seperti BuildingArea, YearBuilt, Landsize, Rooms, Bathroom, Car.
- Goal yang tercapai : perbandingan model yang menunujukkan bahwa Radom Forest sebagai model terbaik untuk prediksi harga.
- Solusi : para agen bisa menggunakan fitur yang prioritas saat menilai properti atau memberikan rekomendasi kepada pembeli. 

**---Ini adalah bagian akhir laporan---**


