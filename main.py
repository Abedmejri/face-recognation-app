import cv2
import numpy as np
import time

# Load the pre-trained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img, faces

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Calculate FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 30.0  # default to 30 fps if cannot be determined
    prev_time = time.time()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame, faces = detect_faces(frame)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps_display = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw labels on faces
        for (x, y, w, h) in faces:
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Video Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the current frame with detected faces
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"frame_{timestamp}.png", frame)
            print(f"Frame saved as frame_{timestamp}.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()