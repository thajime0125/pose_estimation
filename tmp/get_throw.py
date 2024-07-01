import cv2


def get_throw(df, before_frames, after_frames):
    now_round = 1
    before_x = []
    for i, row in df.iterrows():
        if row['round'] != now_round:
            now_round = row['round']
    row['hoge']


def show_with_frames(movie_path):
    cap = cv2.VideoCapture(movie_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 右上にフレーム数を表示
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame, str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    show_with_frames('data/gopro-0507/GX010911.MP4')
    

