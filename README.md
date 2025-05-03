# LAPORAN PROYEK AKHIR KELAS MACHINE LEARNING EXPERT - IDCAMP X DICODING - MUHAMAD JAFAR RAHADIAN 
## <p align="center"> **Sistem Rekomendasi Anime menggunakan Content-Based dan Collaborative-Based(Neural Network) Filtering** </p>

<br>

## **PROJECT OVERVIEW**
<p align="justify">
Di era digital saat ini, industri hiburan mengalami transformasi besar yang juga melanda dunia anime. Jutaan judul anime tersedia di platform streaming, sehingga pengguna sering dihadapkan pada *paradox of choice* atau dilema pilihan berlebih. Tanpa sistem rekomendasi yang efektif, pengguna akan mengalami kesulitan menemukan anime sesuai selera mereka. Oleh karena itu, dibutuhkan sistem rekomendasi yang mampu mempersonalisasi saran anime berdasarkan preferensi pengguna[1]. Implementasi personalisasi telah terbukti meningkatkan keterlibatan dan kepuasan pengguna dalam platform streaming. Bahkan perusahaan global seperti Amazon dan Netflix memanfaatkan sistem rekomendasi personalisasi untuk mempertahankan keunggulan kompetitif dan meningkatkan kepuasan pelanggan[2].
</p>
<p align="justify">
Proyek ini menggunakan *Anime Recommendations Database* dari Kaggle (Cooper Union), yang berisi data interaksi dan preferensi sekitar 76.000 pengguna MyAnimeList[3]. Dengan basis data tersebut, fokus utama adalah mengembangkan sistem rekomendasi anime yang **akurasi** prediksinya tinggi. Keakuratan rekomendasi sangat penting karena rekomendasi yang tepat akan meningkatkan kepuasan dan loyalitas pengguna[4]. Sistem rekomendasi anime yang akurat dapat membantu pengguna mengatasi kelebihan pilihan, memberikan pengalaman menonton yang lebih baik, dan pada akhirnya mendukung pertumbuhan bisnis layanan anime.
</p>

<br>

Daftar Pustaka:<br>
<p align="justify">
[1]	J. Chen, “The Investigation on Anime-Themed Recommendation Systems,” Highlights Sci. Eng. Technol., vol. 81, pp. 121–131, 2024, doi: 10.54097/36drh331.

[2]	J. K. Kim, I. Y. Choi, and Q. Li, “Customer satisfaction of recommender system: Examining accuracy and diversity in several types of recommendation approaches,” Sustain., vol. 13, no. 11, 2021, doi: 10.3390/su13116165.

[3]	Cooper Union, “Anime Recommendations Database,” Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database. [Diakses: 02-Mei-2025].

[4]	T. Silveira, M. Zhang, X. Lin, Y. Liu, and S. Ma, “How good your recommender system is? A survey on evaluations in recommendation,” Int. J. Mach. Learn. Cybern., vol. 10, no. 5, pp. 813–831, 2019, doi: 10.1007/s13042-017-0762-9.
</p>

## **BUSINESS UNDERSTANDING**

### **Problem Steatment**:

- **Kelebihan Pilihan Konten:** Pengguna anime menghadapi jumlah pilihan yang sangat besar (beragam genre dan judul), yang dapat menimbulkan *decision fatigue* dan kebingungan dalam memilih.
- **Kurangnya Personalisasi:** Banyak layanan anime belum memberikan rekomendasi personalisasi yang memadai, sehingga pengguna sulit menemukan anime baru yang sesuai selera mereka.
- **Akurasi Rekomendasi Rendah:** Jika rekomendasi tidak akurat, pengguna bisa kecewa dan kehilangan kepercayaan pada sistem. Rekomendasi yang kurang relevan justru menurunkan kepuasan pengguna.

### **Goals**:

- **Personalisasi Rekomendasi:** Mengembangkan sistem rekomendasi anime yang dapat mempersonalisasi saran berdasarkan preferensi dan riwayat pengguna(Memanfaatkan data interaksi (rating) dan konten judul anime agar platform anime dapat lebih kompetitif dalam meningkatkan retensi pengguna serta memanfaatkan model neural network seperti recommendernet untuk melakukan rekomendasi).
- **Meningkatkan Akurasi:** Menjamin tingkat akurasi tinggi pada rekomendasi untuk memaksimalkan kepuasan dan keterlibatan pengguna.
- **Mengurangi Dampak *Paradox of Choice*:** Menyajikan daftar anime terkurasi yang relevan bagi setiap pengguna, sehingga mengurangi kebingungan akibat banyaknya pilihan.

### **Solution statements**:

- **Collaborative Filtering (RecommenderNet):** Pendekatan ini memanfaatkan data interaksi (rating) pengguna dengan anime untuk menemukan pola kesamaan antar pengguna dan anime. Model *RecommenderNet* berbasis *neural network* (Keras) akan dibuat, di mana masing-masing pengguna dan anime direpresentasikan sebagai vektor *embedding*, lalu hasil *dot product* dilengkapi bias untuk memprediksi rating. Metode ini mirip dengan contoh yang diberikan Keras untuk rekomendasi film menggunakan dataset MovieLens. Pendekatan kolaboratif ini diharapkan menangkap preferensi kolektif pengguna sehingga mampu memberikan rekomendasi yang relevan dan akurat.
- **Content-Based Filtering (TF-IDF pada Judul):** Pendekatan ini menggunakan konten anime — dalam hal ini judul anime — untuk menghitung kemiripan antar item. Setiap judul diubah menjadi vektor fitur menggunakan *TF-IDF* (Term Frequency–Inverse Document Frequency), sehingga kata-kata unik dalam judul memiliki bobot lebih tinggi. Rekomendasi dibuat dengan mencari anime lain yang memiliki kemiripan kosinus tinggi berdasarkan bobot TF-IDF kata di judul. Dengan demikian, sistem dapat merekomendasikan anime baru yang kata-kata judulnya mirip dengan judul anime yang disukai pengguna.
  
**Untuk instalasi API, Framework, ataupun Library dapat dilakukan melalui file requirements.txt*

## **DATA UNDERSTANDING**
Dataset yang digunakan dalam proyek ini merupakan dataset sistem rekomendasi anime dengan hasil 2 file csv yaitu `anime.csv` dan `rating.csv`.

### **Sumber Data**
<p align="justify">
Dataset yang digunakan dalam proyek ini merupakan dataset sistem rekomendasi anime (kartun dari negara jepang jika anda tidak paham apa itu anime) didapatkan menggunakan API `kagglehub` dengan hasil 2 file csv yaitu anime.csv dan rating.csv, dataset tersebut dapat anda jumpai melalui website: 
</p>
https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

### **Feature pada Dataset `anime.csv` dan `rating.csv`**
Berikut adalah tabel dari dataset `anime.csv`:

|       | anime_id |                                              name |                                             genre |  type | episodes | rating | members |
|------:|---------:|--------------------------------------------------:|--------------------------------------------------:|------:|---------:|-------:|--------:|
|   0   |    32281 |                                    Kimi no Na wa. |              Drama, Romance, School, Supernatural | Movie |        1 |   9.37 |  200630 |
|   1   |     5114 |                  Fullmetal Alchemist: Brotherhood | Action, Adventure, Drama, Fantasy, Magic, Mili... |    TV |       64 |   9.26 |  793665 |
|   2   |    28977 |                                          Gintama° | Action, Comedy, Historical, Parody, Samurai, S... |    TV |       51 |   9.25 |  114262 |
|   3   |     9253 |                                       Steins;Gate |                                  Sci-Fi, Thriller |    TV |       24 |   9.17 |  673572 |
|   4   |     9969 |                                     Gintama&#039; | Action, Comedy, Historical, Parody, Samurai, S... |    TV |       51 |   9.16 |  151266 |
|  ...  |      ... |                                               ... |                                               ... |   ... |      ... |    ... |     ... |
| 12289 |     9316 |      Toushindai My Lover: Minami tai Mecha-Minami |                                            Hentai |   OVA |        1 |   4.15 |     211 |
| 12290 |     5543 |                                       Under World |                                            Hentai |   OVA |        1 |   4.28 |     183 |
| 12291 |     5621 |                    Violence Gekiga David no Hoshi |                                            Hentai |   OVA |        4 |   4.88 |     219 |
| 12292 |     6133 | Violence Gekiga Shin David no Hoshi: Inma Dens... |                                            Hentai |   OVA |        1 |   4.98 |     175 |
| 12293 |    26081 |                  Yasuji no Pornorama: Yacchimae!! |                                            Hentai | Movie |        1 |   5.46 |     142 |

Berikut adalah tabel dari dataset `rating.csv`:

|         | user_id | anime_id | rating |
|--------:|--------:|---------:|-------:|
|    0    |       1 |       20 |     -1 |
|    1    |       1 |       24 |     -1 |
|    2    |       1 |       79 |     -1 |
|    3    |       1 |      226 |     -1 |
|    4    |       1 |      241 |     -1 |
|   ...   |     ... |      ... |    ... |
| 7813732 |   73515 |    16512 |      7 |
| 7813733 |   73515 |    17187 |      9 |
| 7813734 |   73515 |    22145 |     10 |
| 7813735 |   73516 |      790 |      9 |
| 7813736 |   73516 |     8074 |      9 |

Berdasarkan kedua tabel diatas, dataset `anime.csv` dan `rating.csv` berisi  informasi mengenai rating anime dari total pengguna kurang lebih sebesar 70.000 pengguna pada 12.294 anime. Untuk setiap dataset memiliki beberapa feature, diantaranya:
  
1. Dataset anime.csv:
    - anime_id: ID unik dari myanimelist.net yang mengidentifikasi sebuah anime.
    - name: nama lengkap anime.
    - genre: daftar genre yang dipisahkan koma untuk anime ini.
    - type: MOVIE, TV, OVA, dll.
    - episodes: jumlah episode dalam acara ini. (1 jika film).
    - rating: peringkat rata-rata dari 10 untuk anime ini.
    - members: jumlah anggota komunitas yang tergabung dalam "grup" anime ini.

2. Dataset rating.csv:
    - user_id: id pengguna yang dibuat secara acak dan tidak dapat diidentifikasi.
    - anime_id: anime yang telah diberi peringkat oleh pengguna ini.
    - rating: peringkat dari 10 yang diberikan oleh pengguna ini (-1 jika pengguna menontonnya tetapi tidak memberikan peringkat).

### **Informasi Dataset**
Kemudian adapun informasi dari kedua dataset yang dapat dilihat pada penjelasan berikut:

1. anime.csv
   
  ```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 12294 entries, 0 to 12293
  Data columns (total 7 columns):
   #   Column    Non-Null Count  Dtype  
  ---  ------    --------------  -----  
   0   anime_id  12294 non-null  int64  
   1   name      12294 non-null  object 
   2   genre     12232 non-null  object 
   3   type      12269 non-null  object 
   4   episodes  12294 non-null  object 
   5   rating    12064 non-null  float64
   6   members   12294 non-null  int64  
  dtypes: float64(1), int64(2), object(4)
  memory usage: 672.5+ KB
  ```
DataFrame ini terdiri dari 12.294 entri (indeks 0 hingga 12293) dengan 7 kolom, yaitu `anime_id` (int64, 12.294 non-  null), `name` (object, 12.294 non-null), `genre` (object, 12.232 non-null), `type` (object, 12.269 non-null), `episodes` (object, mungkin berisi nilai seperti "Unknown"), `rating` (float64, 12.064 non-null), dan `members` (int64, 12.294 non-null). Terdapat missing values pada kolom `genre` (62), `type` (25), dan `rating` (230). Tipe data terdiri dari 1 kolom float64 (`rating`), 2 kolom int64 (`anime_id` dan `members`), serta 4 kolom object (`name`, `genre`, `type`, `episodes`), dengan penggunaan memori sekitar 672.5+ KB. Kolom `episodes` disimpan sebagai object kemungkinan karena berisi nilai non-numerik atau format bervariasi, sementara `members` dan `anime_id` sudah bersih tanpa missing values.

Insight:
  - Terdapat missing values pada kolom genre (62), type (25), dan rating (230)- Kolom episodes disimpan sebagai object (bukan numerik), mungkin karena:
  - Berisi nilai non-numerik seperti "Unknown"
  - Format bervariasi (e.g., "12", "12+")
  - Kolom members dan anime_id sudah bersih tanpa missing values

2. rating.csv
   
  ```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 7813737 entries, 0 to 7813736
  Data columns (total 3 columns):
   #   Column    Dtype
  ---  ------    -----
   0   user_id   int64
   1   anime_id  int64
   2   rating    int64
  dtypes: int64(3)
  memory usage: 178.8 MB
  ```
DataFrame rating memiliki ukuran sangat besar dengan **7.813.737 entri** (indeks 0 hingga 7.813.736) dan terdiri dari **3 kolom** bertipe integer (`int64`), yaitu `user_id` (ID unik pengguna), `anime_id` (ID unik anime), dan `rating` (nilai rating yang diberikan pengguna), semuanya dalam format numerik tanpa missing values.

Insight:
   - Tidak ada missing values (semua kolom menunjukkan jumlah entri yang sama dengan total entries)
   - Semua data berbentuk numerik murni (terlihat dari Dtype yang semuanya int64)
   - Ukuran memori yang digunakan cukup besar: **178.8 MB**

### **Statistik Deskriptif Kedua Dataset**
Berikut adalah tabel deskripsi dataset `anime_df` non objek:

|       |     anime_id |       rating |      members |
|------:|-------------:|-------------:|-------------:|
| count | 12294.000000 | 12064.000000 | 1.229400e+04 |
|  mean | 14058.221653 |     6.473902 | 1.807134e+04 |
|  std  | 11455.294701 |     1.026746 | 5.482068e+04 |
|  min  |     1.000000 |     1.670000 | 5.000000e+00 |
|  25%  |  3484.250000 |     5.880000 | 2.250000e+02 |
|  50%  | 10260.500000 |     6.570000 | 1.550000e+03 |
|  75%  | 24794.500000 |     7.180000 | 9.437000e+03 |
|  max  | 34527.000000 |    10.000000 | 1.013917e+06 |

Penjelasan mengenai tabel diatas:
1. **Keseluruhan Data**:
   - Analisis ini hanya mencakup 3 kolom numerik dari DataFrame: `anime_id`, `rating`, dan `members`
   - Kolom non-numerik (seperti `name`, `genre`, `type`, & `episode` ) tidak ditampilkan dalam output ini

2. **Statistik untuk Setiap Kolom**:

   a. **anime_id**:
   - `count`: 12,294 entries (lengkap tanpa missing values)
   - `mean`: Rata-rata ID anime 14,058.22
   - `std`: Deviasi standard tinggi (11,455.29) menunjukkan sebaran ID yang lebar
   - `min`: ID terkecil 1
   - `max`: ID terbesar 34,527

   b. **rating**:
   - `count`: 12,064 entries (ada 230 missing values, sesuai info sebelumnya)
   - `mean`: Rating rata-rata 6.47 (skala 1-10)
   - `50%`: Median 6.57 (sedikit lebih tinggi dari mean)
   - `min`: Rating terendah 1.67
   - `max`: Rating tertinggi 10 (sempurna)

   c. **members**:
   - `count`: 12,294 entries (lengkap)
   - `mean`: Rata-rata 18,071 member per anime
   - `std`: Deviasi standard sangat besar (54,820.68)
   - `min`: Anime dengan member paling sedikit (5 member)
   - `max`: Anime paling populer (1,013,917 member)

3. **Insight Penting**:
   - Distribusi rating cenderung normal (mean dan median dekat)
   - Angka members sangat skewed (perbedaan besar antara mean dan max)
   - 25% anime memiliki kurang dari 225 member
   - 50% anime memiliki kurang dari 1,550 member
   - Hanya 25% anime yang memiliki lebih dari 9,437 member

Kemudian, berikut adalah tabel deskripsi dataset `anime_df` bertipe objek:

|        |                    name |  genre |  type | episodes |
|-------:|------------------------:|-------:|------:|---------:|
|  count |                   12294 |  12232 | 12269 |    12294 |
| unique |                   12292 |   3264 |     6 |      187 |
|   top  | Shi Wan Ge Leng Xiaohua | Hentai |    TV |        1 |
|  freq  |                       2 |    823 |  3787 |     5677 |

Penjelasan mengenai tabel diatas:
1. **Metode Analisis**:
   - `describe(include=object)` fokus pada kolom dengan tipe data object/string
   - Menghasilkan statistik deskriptif khusus untuk data kategorikal/textual

2. **Statistik untuk Setiap Kolom**:

   a. **name (Judul Anime)**:
   - `count`: 12,294 entries (lengkap, tanpa missing values)
   - `unique`: 12,292 judul unik
   - `top`: "Saru Kani Gassen" adalah judul paling umum
   - `freq`: 2 (artinya ada 2 anime dengan judul ini)
   - *Insight*: Hampir semua judul unik (hanya 1 duplikat)

   b. **genre**:
   - `count`: 12,232 (ada 62 missing values)
   - `unique`: 3,264 kombinasi genre unik
   - `top`: "Hentai" adalah genre paling umum
   - `freq`: 823 anime bergenre ini
   - *Insight*: Variasi genre sangat banyak dengan dominasi konten dewasa

   c. **type (Tipe Anime)**:
   - `count`: 12,269 (ada 25 missing values)
   - `unique`: 6 kategori unik
   - `top`: "TV" adalah tipe paling populer
   - `freq`: 3,787 anime (≈30% dari total)
   - *Insight*: Mayoritas anime berupa serial TV

   d. **episodes**:
   - `count`: 12,294 (lengkap)
   - `unique`: 187 nilai unik
   - `top`: "1" adalah nilai paling umum
   - `freq`: 5,677 anime (≈46% dari total)
   - *Insight*: Banyak anime single-episode (OVA/Movie)

3. **Insight Penting**:
   - **Distribusi Tidak Seimbang**: Beberapa kategori mendominasi (TV, single-episode)

Terakhir, tabel deksripsi untuk dataset `rating_df`:

|       |      user_id |     anime_id |        rating |
|------:|-------------:|-------------:|--------------:|
| count | 7.813737e+06 | 7.813737e+06 |  7.813737e+06 |
|  mean | 3.672796e+04 | 8.909072e+03 |  6.144030e+00 |
|  std  | 2.099795e+04 | 8.883950e+03 |  3.727800e+00 |
|  min  | 1.000000e+00 | 1.000000e+00 | -1.000000e+00 |
|  25%  | 1.897400e+04 | 1.240000e+03 |  6.000000e+00 |
|  50%  | 3.679100e+04 | 6.213000e+03 |  7.000000e+00 |
|  75%  | 5.475700e+04 | 1.409300e+04 |  9.000000e+00 |
|  max  | 7.351600e+04 | 3.451900e+04 |  1.000000e+01 |

Penjelasan mengenai tabel diatas:

**1. Statistik Dasar (Count, Mean, Std Dev):**
- Dataset berisi **7,813,737 rating** dari pengguna (count sama untuk semua kolom)
- **user_id:**
  - Rata-rata ID user: 36,728 (mean)
  - Distribusi merata (std dev 20,998) menunjukkan penyebaran user yang baik
- **anime_id:**
  - Rata-rata ID anime: 8,909 (mean)
  - Std dev 8,884 menunjukkan distribusi yang cukup merata
- **rating:**
  - Skor rata-rata: 6.14 (skala -1 sampai 10)
  - Std dev 3.73 menunjukkan variasi rating yang besar

**2. Nilai Ekstrim (Min/Max):**
- **user_id:**
  - Range 1 sampai 73,516 (total ≈73.5k user unik)
- **anime_id:**
  - Range 1 sampai 34,519 (sesuai dengan anime_df)
- **rating:**
  - Nilai minimum -1 (artinya "user tidak memberi rating")
  - Maksimum 10 (rating sempurna)

**3. Distribusi (Percentil 25/50/75):**
- **user_id:**
  - 25% user berada di bawah ID 18,974
  - Median di ID 36,791
  - 75% di bawah ID 54,757
- **anime_id:**
  - 25% anime memiliki ID <1,240
  - Median di ID 6,213
  - 75% di bawah ID 14,093
- **rating:**
  - 25% rating ≤6
  - Median rating 7
  - 75% rating ≤9

**4. Insight Penting:**
  - Mayoritas rating (50%) di atas mean (6.14 vs median 7)
  - Ada kecenderungan rating positif (75% data ≤9)
  - Cukup banyak user aktif (median 36k ID)
  - Distribusi rating cenderung ke nilai positif

### **Kondisi Data (Duplikat dan Missing Value)**
Pertama, memeriksa kemungkinan data duplikat pada dataset `anime_df`, berikut adalah hasilnya:
```
Duplicate Rows in anime_df:
anime_id	name	genre	type	episodes	rating	members

Duplicate 'anime_id' in anime_df:
anime_id	name	genre	type	episodes	rating	members
```
Insight:
- Tidak ada nilai `anime_id` yang terduplikasi
- `anime_id` valid sebagai primary key

Kemudian, memeriksa kemungkinan data duplikat pada dataset `rating_df`, berikut adalah hasilnya:

Duplicate row in rating_df:

|         | user_id | anime_id | rating |
|--------:|--------:|---------:|-------:|
| 4499316 |   42653 |    16498 |      8 |

Duplicate `user_id` and `anime_id` combinations in rating_df:

|         | user_id | anime_id | rating |
|--------:|--------:|---------:|-------:|
| 4499286 |   42653 |     1575 |      6 |
| 4499288 |   42653 |     2001 |     10 |
| 4499307 |   42653 |    11757 |      5 |
| 4499316 |   42653 |    16498 |      8 |
| 4499320 |   42653 |    20507 |      9 |
| 4499325 |   42653 |    22319 |      6 |
| 4499326 |   42653 |    23283 |      9 |

Insight:
- Data rating mengandung ~0.00009% duplikat penuh (1 dari 7.8 juta)
- Terdapat 7 kasus rating ganda dari user yang sama ke anime yang sama

Kemudian, memeriksa missing value pada dataset `anime_df`, berikut hasilnya:

|          |   0 |
|---------:|----:|
| anime_id |   0 |
|   name   |   0 |
|   genre  |  62 |
|   type   |  25 |
| episodes |   0 |
|  rating  | 230 |
|  members |   0 |

Insight:
- **`genre`**: 62 entri kosong  
  → ~0.5% dari total data (62/12294)
- **`type`**: 25 entri kosong  
  → Kategori anime tidak tercatat untuk sebagian kecil data
- **`rating`**: 230 entri kosong  
  → ~1.87% anime tidak memiliki rating
- Kolom identifier (`anime_id`, `name`) sudah bersih

Terakhir, memeriksa missing value pada dataset `rating_df`, berikut hasilnya:

|          | 0 |
|---------:|--:|
|  user_id | 0 |
| anime_id | 0 |
|  rating  | 0 |

Insight:
Dapat disimpulkan bahwa dataset `rating_df` tidak memiliki missing value.

### **Distribusi Rating**


### **Distribusi Kategori Anime**
### **Analisis: 10 Anime dengan Member Paling Banyak**
### **Analisis: 10 Anime dengan Member Paling Sedikit**
### **Analisis: 10 Anime dengan Rating Paling Tinggi**
### **Analisis: 10 Anime dengan Rating Paling Rendah**
### **Analisis: Sebaran Genre Anime**
### **Wordcloud Genre Anime**

## **DATA PREPARATION**
### **Data Cleaning (Menghapus Missing Value, Duplikat, dan Outlier)**
### **Encoding Data - Content-Based Filtering**
### **Dataframe Cosine Similarity**
### **Menghapus Karakter Spesial pada Cosine Similarity**
### **Pra-pemrosesan Teks Genre Anime**
### **Normalisasi Rating**
### **Encoding Data - Collaborative Filtering**
### **Encoding Data - Memilih 150.000 Data**
### **Split Dataset**
### **Restrukturisasi Data**

## **MODELING**
### **Content-Based Filtering**
### **Collaborative Filtering**

## **EVALUATION**
### **Penjelasan Matriks Evaluasi MSE**
### **Penjelasan Matriks Evaluasi RMSE**
### **Evaluasi Model Collaborative (RecommenderNet)**
### **Kesimpulan**
