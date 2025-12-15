import cv2
import numpy as np
import os
import mediapipe as mp
import time

# --- Konfigurasi ---
DATA_DIR = "dataset_skeleton"

CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
SAMPLES_PER_CLASS = 50    # Jumlah Sampel Per Huf
IMG_SIZE = 128             # Ukuran Citra Skeleton Akhir (128x128)

# --- Inisialisasi Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi deteksi tangan
# static_image_mode=false: Lebih cepat untuk video karena menganggap antar frame saling berkaitan
# max_num_hands=2: Deteksi 2 tangan untuk Abjad BISINDO
# min_detection_confidence=0.5: Ambang batas keyakinan agar dianggap tangan
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Persiapan Folder
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for label in CLASSES:
    path = os.path.join(DATA_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)

# --- Mulai Pengambilan Data ---
cap = cv2.VideoCapture(0) # 0 Adalah ID bawaan laptop/PC

print("Panduan Penggunaan: ")
print("1. Tunjukkan pose tangan sesuai instruksi di layar.")
print("2. Tekan 'R' untuk mulai merekam data kelas.")
print("3. Tekan 'SPACE' untuk pause/resume perekaman.")
print("4. Tekan 'D' untuk delete frame terakhir.")
print("5. Tekan 'N' untuk skip ke kelas berikutnya.")
print("6. Tekan 'Q' untuk keluar paksa.")

for label in CLASSES:
    count = 0
    print(f"\n>>> SIAP MEREKAM KELAS: {label}")

    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca kamera!")
            break
        
        # Flip horizontal agar seperti cermin (lebih natural bagi pengguna)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Konversi BGR ke RGB (Mediapipe buth RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses deteksi tangan
        result = hands.process(rgb_frame)

        # Siapkan kanvas hitam kosong seukuran frame kamera
        skeleton_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Jika tangan terdeteksi
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 1. Gambar skeleton di frame asli (untuk feedback visual ke pengguna)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 2. INOVASI UTAMA: Gambar skeleton di kanvas hitam
                # Atur Warna Putih (255, 255, 255) dan ketebalan garis
                mp_drawing.draw_landmarks(
                    skeleton_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), # Titik sendi
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2) # Garis Penghubung
                )

        # --- LOGIKA PEREKAMAN ---
        key = cv2.waitKey(1)

        if key == ord('r') and not recording:
            start_timer = time.time()
            countdown_duration = 3

            while (time.time() - start_timer) < countdown_duration:
                ret, frame_countdown = cap.read()
                if not ret: break

                # Proses frame agar pengguna bisa mirroring
                frame_countdown = cv2.flip(frame_countdown, 1)

                # Hitung Sisa Waktu
                elapsed = time.time() - start_timer
                remaining_time = int(np.ceil(countdown_duration - elapsed))

                # Tampilkan Angka Besar di Tengah Layar
                cv2.putText(frame_countdown, str(remaining_time), (int(w/2)-50, int(h/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 10)
                cv2.putText(frame_countdown, f"Siap Pose {label}...", (int(w/2)-200, int(h/2)+100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Tampilkan Live Preview
                # Tampilkan skeleton kosong di kanan agar layout tetap sama
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                combined_preview = np.hstack((frame_countdown, blank))

                cv2.imshow("Data Collection: Asli vs Skeleton", combined_preview)

                # waitKey(1) membuat video tetap jalan dan tidak macet
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Countdown selesai, mulai perekaman
            print(f"Mulai merekam data BISINDO '{label}'...")
            recording = True

        if recording:
            # Memastikan ada minimal 1 tangan terdeteksi sebelum menyimpan
            if result.multi_hand_landmarks:
                # Resize citra skeleton hitam putih ke ukuran target (64x64)
                final_img = cv2.resize(skeleton_img, (IMG_SIZE, IMG_SIZE))

                # Simpan file
                save_path = os.path.join(DATA_DIR, label, f"{count}.jpg")
                cv2.imwrite(save_path, final_img)
                count += 1
                # Tampilkan progress di console
                print(f"Terekam {count}/{SAMPLES_PER_CLASS}", end='\r')

                time.sleep(0.1) # Memberi waktu pengguna untuk sedikit menggoyangkan tangan
            
            else:
                print("Tangan tidak terdeteksi! Frame dilewati.", end='\r')

        # Tampilkan Status di Layar
        cv2.putText(frame, f"Target: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Progress: {count}/{SAMPLES_PER_CLASS}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if recording:
            cv2.putText(frame, "MEREKAM...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tekan 'R' untuk mulai", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Tampilkan Preview: Kiri (Kamera Asli), Kanan (Hasil Konversi)
        # Resize skeleton_img agar muat di layar preview
        preview_skeleton = cv2.resize(skeleton_img, (w, h))
        combined_view = np.hstack((frame, preview_skeleton))

        cv2.imshow("Data Collection: Asli vs Skeleton", combined_view)

        # Cek jika sudah selesai satu kelas
        if recording and count >= SAMPLES_PER_CLASS:
            print(f"\nSelesai merekam kelas {label}!")
            recording = False
            break # Lanjut ke huruf berikutnya di loop luar
        
        if recording:
            if key == ord(' '):  # SPACE untuk pause
                recording = False
                print(f"Perekaman DIJEDA. Tekan R untuk lanjut atau N untuk skip kelas.")
            
            elif key == ord('d'):  # D untuk delete last frame
                if count > 0:
                    last_file = os.path.join(DATA_DIR, label, f"{count-1}.jpg")
                    if os.path.exists(last_file):
                        os.remove(last_file)
                        count -= 1
                        print(f"Frame terakhir dihapus. Progress: {count}/{SAMPLES_PER_CLASS}")
        
        elif key == ord('n'):  # N untuk skip kelas
            print(f"Melewati kelas '{label}' ({count}/{SAMPLES_PER_CLASS} sampel).")
            break

        if key == ord('q'):
            print("\nKeluar Paksa.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("\nSemua Kelas Selesai Direkam!")
cap.release()
cv2.destroyAllWindows()
