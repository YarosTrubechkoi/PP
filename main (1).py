import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import easyocr
os.makedirs('D:/TRASH!/uga', exist_ok=True)
OUTPUT_VIDEO_PATH = os.path.join('D:/TRASH!/uga', 'proce_video.avi')
model = YOLO('carplate_model/weights/best.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
reader = easyocr.Reader(['ru'], gpu=torch.cuda.is_available())
cap = cv2.VideoCapture('videos/sample_short.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30, (1920, 1080))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

    for x1, y1, x2, y2 in boxes:
        plate_img = frame[y1:y2, x1:x2]
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        result = reader.readtext(
            plate_gray,
            allowlist='АВЕКМНОРСТУХ0123456789',
            contrast_ths=8,
            adjust_contrast=0.85,
            add_margin=0.015,
            width_ths=20,
            decoder='beamsearch',
            text_threshold=0.1,
            batch_size=8,
            beamWidth=32
        )

        plate_text = ''
        for _, text, score in result:
            if score > 0.5:
                plate_text += text.upper().replace(' ', '')

        # Рисуем результаты на кадре
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Записываем кадр прямо в видео
    video_out.write(frame)
    frame_count += 1

cap.release()
video_out.release()
cv2.destroyAllWindows()
print(f'[✅] Готово! Обработано {frame_count} кадров.')
print(f'[📁] Видео сохранено в: {OUTPUT_VIDEO_PATH}')