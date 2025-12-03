# UAS-PCD
Project Ujian Akhir Mata Kuliah Pengolahan Citra Digital

Lib: tensorflow mediapipe opencv-python numpy matplotlib

Tahap 1: Data Collection (Pembuatan Dataset Citra Skeleton)

Tujuan Tahap ini:
1. Mendeteksi Tangan pengguna real-time menggunakan Mediapipe.
2. Mengambil koordinat landmark tangan.
3. Menggambar ulang landmark tersebut menjadi garis putih di atas latar belakang hitam (Citra Skeleton).
4. Menyimpan Citra Skeleton tersebut ke dalam folder yang terorganisir berdasarkan abjad (A, B, C, dst.) saat tombol tertentu ditekan.