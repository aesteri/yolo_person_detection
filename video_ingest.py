import cv2
import random
import sys
from ultralytics import YOLO

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def process_video(video_capture, model):
    """
    Process the video frame by frame, display the output, and handle user exit.
    """
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        results = model.track(frame, stream=True, persist=True)

        for result in results:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])
                    class_name = class_names[cls]

                    if class_name != "person":
                        continue

                    conf = float(box.conf[0])

                    colour = getColours(cls)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

def main():
    """
    Main function to initialize model, video capture, and start processing.
    """
    if len(sys.argv) < 2:
        print("Please provide video file path")

    video_path = sys.argv[1]

    print("Loading YOLO model...")
    yolo = YOLO("yolov8s.pt")

    print(f"Opening video file: {video_path}")
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file at {video_path}")

    process_video(video_capture, yolo)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()