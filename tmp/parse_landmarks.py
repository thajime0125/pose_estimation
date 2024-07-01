import datetime
import os

import cv2
import mediapipe as mp
import pandas as pd

from tmp.detect_by_mp import detect_landmarks


def create_mst_data(data_dir='data/'):
    df = pd.DataFrame(columns=['movie_file', 'csv_file', 'fps', 'frame_count', 'duration', 'date', 'player'])
    df.to_csv('mst_data/mst_game.csv', index=False)


def regist_mst_data(data_dir='data/', csv_dir='csv_data/'):
    # Get the list of all files in directory tree at given path
    list_of_files = list()
    file_data = pd.read_csv('mst_data/mst_game.csv')
    data = []
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for file in filenames:
            if file.endswith(('.mp4', '.MP4')):
                list_of_files.append(os.path.join(dirpath, file))
    for file in list_of_files:
        if file_data[file_data['movie_file'] == file].empty:
            fps, frame_count, duration = get_video_info(file)
            row = {'movie_file': file, 'csv_file': csv_dir+file.split('/')[-1].replace('.MP4', '.csv'), 'fps': fps, 'frame_count': frame_count, 'duration': duration, 'date': datetime.datetime.now().date(), 'player': ''}
            # row = {'file': file, 'fps': fps, 'frame_count': frame_count, 'date': datetime.datetime(2024, 5, 7).date(), 'player': ''}
            data.append(row)
    df = pd.concat([file_data, pd.DataFrame(data)])
    df.to_csv('mst_data/mst_game.csv', index=False)


def get_video_info(file):
    mov = cv2.VideoCapture(file)
    fps = round(mov.get(cv2.CAP_PROP_FPS))
    frame_count = int(mov.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(frame_count / fps)
    mov.release()
    return fps, frame_count, duration


def parse_landmarks(landmarks, round, timestamp, throw = 0):
    landmark_dict = {}
    landmark_dict['round'] = round
    landmark_dict['timestamp'] = timestamp
    landmark_dict['throw'] = throw
    for i in range(33):
        landmark_dict[f'l{i}x'] = landmarks[0][i].x
        landmark_dict[f'l{i}y'] = landmarks[0][i].y
        landmark_dict[f'l{i}z'] = landmarks[0][i].z
    return landmark_dict

def video_to_parse_csv(video_path, model_path, csv_path):

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )

    landmark_dict_list = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        video = cv2.VideoCapture(video_path)
        round = 0
        round_count = 0
        is_round = False

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            frame_timestamp_ms = int(timestamp)

            # Process the frame using the pose landmarker.
            pose_landmarker_result = detect_landmarks(frame, frame_timestamp_ms, landmarker)

            if pose_landmarker_result.pose_world_landmarks and pose_landmarker_result.pose_world_landmarks[0][32].visibility > 0.9:
                if not is_round:
                    round_count += 1
                    if round_count > 5:
                        round_count = 0
                        is_round = True
                        round += 1
                        print(f"Round {round}")
                else:
                    landmark_dict = parse_landmarks(pose_landmarker_result.pose_world_landmarks, round, frame_timestamp_ms)
                    landmark_dict_list.append(landmark_dict)

            else:
                if is_round:
                    round_count += 1
                    if round_count > 5:
                        is_round = False
                        round_count = 0
                        print("Round end")
                        if round == 8:
                            break
        
        df = pd.DataFrame(landmark_dict_list)
        df.to_csv(csv_path, index=False)

def get_landmarks_with_frame_count(video_path, model_path, start_frame, end_frame, round, throw) -> pd.DataFrame:
    
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
    
        # Create a pose landmarker instance with the video mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
        )
    
        landmark_dict_list = []
    
        with PoseLandmarker.create_from_options(options) as landmarker:
            video = cv2.VideoCapture(video_path)
    
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
    
                timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
                frame_timestamp_ms = int(timestamp)
                frame_count = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    
                if frame_count < start_frame:
                    continue
                if frame_count > end_frame:
                    break
    
                # Process the frame using the pose landmarker.
                pose_landmarker_result = detect_landmarks(frame, frame_timestamp_ms, landmarker)
                landmark_dict = parse_landmarks(pose_landmarker_result.pose_world_landmarks, round, frame_timestamp_ms, throw)
                landmark_dict_list.append(landmark_dict)
    
            
            df = pd.DataFrame(landmark_dict_list)
            return df
