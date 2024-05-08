
from pprint import pprint

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MODEL_PATH = "models/pose_landmarker_full.task"
VIDEO_PATH = "data/gopro-0507/GX010911.MP4"


def draw_landmarks_on_image(cv2_image, detection_result):
  rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def detect_landmarks(cv2_image, frame_timestamp_ms):
   # Convert the frame to RGB using OpenCV’s cvtColor() function.
    frame = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert the frame to MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Process the frame using the pose landmarker.
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    return pose_landmarker_result



BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    video = cv2.VideoCapture(VIDEO_PATH)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamp_ms = int(timestamp)

        # Process the frame using the pose landmarker.
        pose_landmarker_result = detect_landmarks(frame, frame_timestamp_ms)

        # Print the landmarks detected in the frame.
        if pose_landmarker_result.pose_landmarks:
            pprint(pose_landmarker_result.pose_landmarks[0][32].visibility)

        annotated_image = draw_landmarks_on_image(frame, pose_landmarker_result)
        cv2.imshow('Mediapipe', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
