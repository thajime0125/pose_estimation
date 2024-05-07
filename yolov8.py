import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

video_path = "data/20230514192027_cut.mp4"
video = cv2.VideoCapture(video_path)

video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_hight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_filename = "output_movie_yolo.mp4"
output_file = cv2.VideoWriter(
    output_filename, cv2.VideoWriter_fourcc(*"MP4V"), 30, (video_width, video_hight)
)

while video.isOpened():
    success, frame = video.read()
    if success:
        results = model(frame)

        annotatedFrame = results[0].plot()
        # cv2.imshow("YOLOv8 Inference", annotatedFrame)
        output_file.write(annotatedFrame)

        # names = results[0].names
        # classes = results[0].boxes.cls
        # boxes = results[0].boxes

        # for box, cls in zip(boxes, classes):
        #     name = names[int(cls)]
        #     x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]

        # if len(results[0].keypoints) == 0:
        #     continue

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        break


video.release()
output_file.release()
