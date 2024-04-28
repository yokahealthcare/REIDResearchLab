import os

import cv2

from yolo_detector import YoloDetector


def save_frame_according_to_id(frame, person_id):
    folder_name = f"asset/image/images_by_id/{person_id}"

    folder_exist = os.path.exists(folder_name)
    if not folder_exist:
        os.mkdir(folder_name)
        largest_number = 0
    else:
        # List all file inside folder
        files = os.listdir(folder_name)
        numbers = [int(filename[:-4]) for filename in files]
        largest_number = max(numbers)

    saved_filename = f"{largest_number + 1}"
    cv2.imwrite(f"asset/image/images_by_id/{person_id}/{saved_filename}.jpg", frame)


if __name__ == '__main__':
    yolo = YoloDetector("asset/model/yolo/yolov8l.pt")

    cap = cv2.VideoCapture("asset/video/sample#1.mp4")
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        result = yolo.track(frame, classes=[0])
        if result.boxes.id is None:
            continue
        ids = result.boxes.id.clone().tolist()
        boxes = result.boxes.xyxy.clone().tolist()
        for id, (x1, y1, x2, y2) in zip(ids, boxes):
            id, x1, y1, x2, y2 = int(id), int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))
            cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            person = frame[y1:y2, x1:x2]
            save_frame_according_to_id(person, id)

        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
