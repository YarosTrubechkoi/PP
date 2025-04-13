from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import os
import torch
model=YOLO('carplate_model/weights/best.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
video_name = 'mygeneratedvideo.avi'
cap=cv2.VideoCapture('videos/sample_short.mp4')
images=[]
path = "D:/TRASH!/uga"
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 3, (1920, 1080))
os.chdir(path)

while cap.isOpened():
    ret,frame=cap.read()
    if ret==False:
        break
    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

    # Вырезаем номера по координатам
    for x, y, w, h in boxes:
        carplate_img = frame[y:h, x:w]
        carplate_img_gray = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2GRAY)

        # Считываем номера
        reader = easyocr.Reader(['ru'], gpu=True)
        data = reader.readtext(carplate_img_gray, allowlist='АВEКМНОРСТУХ0123456789',
                               contrast_ths=8, adjust_contrast=0.85, add_margin=0.015, width_ths=20,
                               decoder='beamsearch', text_threshold=0.1, batch_size=8, beamWidth=32)
        text_full = ''
        for l in data:
            bbox, text, score = l
            if score > 0.5:
                text_full += text
                text_full = text_full.upper().replace(' ', '')

        # Отрисовываем всю инфу на изображении
        final_img = cv2.putText(frame, text_full, (x, y - 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
        final_img = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        final_img = cv2.resize(final_img, (final_img.shape[1] // 2, final_img.shape[0] // 2))
        images.append(final_img)
        # Video writer to create .avi filд

        # Appending images to video
for image in images:
    video.write(image)

video.release()
cap.release()
cv2.destroyAllWindows()



