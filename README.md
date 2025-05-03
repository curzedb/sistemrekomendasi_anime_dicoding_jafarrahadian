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
Dataset yang digunakan dalam proyek ini merupakan dataset sistem rekomendasi anime (kartun dari negara jepang jika anda tidak paham apa itu anime) dengan hasil 2 file csv yaitu `anime.csv` dan `rating.csv`.

### **Sumber Data**
### **Deskripsi Dataset `anime.csv` dan `rating.csv`**
### **Kondisi Data (Duplikat, Missing Value, dan Outlier)**
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
