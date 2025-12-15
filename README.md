# UAS-PCD
Project Ujian Akhir Mata Kuliah Pengolahan Citra Digital

Lib: tensorflow mediapipe opencv-python numpy matplotlib

Tahap 1: Data Collection (Pembuatan Dataset Citra Skeleton)

Tujuan Tahap ini:
1. Mendeteksi Tangan pengguna real-time menggunakan Mediapipe.
2. Mengambil koordinat landmark tangan.
3. Menggambar ulang landmark tersebut menjadi garis putih di atas latar belakang hitam (Citra Skeleton).
4. Menyimpan Citra Skeleton tersebut ke dalam folder yang terorganisir berdasarkan abjad (A, B, C, dst.) saat tombol tertentu ditekan.

 Panduan Penggunaan:  
 1. Tunjukkan pose tangan sesuai instruksi di layar. 
 2. Tekan 'R' untuk mulai merekam data kelas. 
 3. Tekan 'SPACE' untuk pause/resume perekaman. 
 4. Tekan 'D' untuk delete frame terakhir. 
 5. Tekan 'N' untuk skip ke kelas berikutnya. 
 6. Tekan 'Q' untuk keluar paksa. 

 For Run Code:.\.venv\Scripts\Activate.ps1; python Aplikasi_BISINDO\1_collect_data.py
