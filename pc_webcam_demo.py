"""
RoboSignFormer PC Canlı Webcam Demo (ROS2 veya PyTorch gerektirmez, ONNXRuntime ile çalışır).
Kullanım:
    python pc_webcam_demo.py              # Varsayılan USB/Laptop kamerası
    python pc_webcam_demo.py ornek.mp4    # Kayıtlı video dosyası ile test
Kısayollar:
    [m] : Kamera Ayna (Mirror) Modunu Aç/Kapat
    [q] : Çıkış
"""
import os
import sys
import time

# Windows USB kameralarda OpenCV MSMF takılmasını önleyen resmi ayar
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import cv2
from robosignformer.pipeline import RoboSignFormerPipeline


def open_video_source(source=0):
    if isinstance(source, str) and os.path.isfile(source):
        return cv2.VideoCapture(source)

    cam_indices = [int(source)] if str(source).isdigit() else [0, 1]
    if int(cam_indices[0]) == 0:
        cam_indices = [0, 1]

    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for idx in cam_indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # USB kamera sensörünün uyanması için kısa ısınma ve 10 denemeye kadar ilk kare okuma
            time.sleep(0.35)
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"✅ Kamera #{idx} başarıyla açıldı ({frame.shape[1]}x{frame.shape[0]}).")
                    return cap
                time.sleep(0.1)
            cap.release()
    return None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    print("📷 USB Kamera uyandırılıyor...")
    cap = open_video_source(src)
    if cap is None or not cap.isOpened():
        print(
            "\n❌ HATA: USB Kamera açılamadı!\n"
            "Lütfen kameranın başka bir sekmede/uygulamada (Colab, Tarayıcı, Zoom, Kamera) açık olmadığını kontrol edin.\n"
        )
        return

    pipeline = RoboSignFormerPipeline(model_dir="models", conf_threshold=0.50, mirror_camera=False)
    win_name = "RoboSignFormer - Live Turkish Sign Language Recognition"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print("🎥 Pencere açıldı! ([m]: Ayna Modu | [q]: Çıkış)")

    missed_frames = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                missed_frames += 1
                if missed_frames > 15:
                    break
                time.sleep(0.02)
                continue
            missed_frames = 0

            kelime, skor, vis = pipeline.process_frame(frame, draw_hud=True)
            if kelime:
                print(f"🤟 Algılanan İşaret: {kelime.upper()} (%{skor})")

            cv2.imshow(win_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("m"):
                pipeline.mirror_camera = not pipeline.mirror_camera
                print(f"↔️ Ayna Modu: {'AÇIK' if pipeline.mirror_camera else 'KAPALI'}")
    finally:
        pipeline.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
