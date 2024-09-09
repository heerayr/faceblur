import cv2
import numpy as np
import time

# Load the Haar Cascade for face detection
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if cascade.empty():
    print("Error loading cascade file.")
    exit()

# Capture video from the default webcam (0 for internal, 1 for external)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error opening video capture.")
    exit()

# Initialize frame rate calculation
fps = video_capture.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps:.2f}")

# Display instructions
instruction_text = "Press 'q' to exit."

while True:
    # Capture the latest frame from the video
    check, frame = video_capture.read()

    if not check:
        print("Error capturing frame.")
        break

    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        # Apply Gaussian blur to the face region
        face_region = frame[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_region, (35, 35), 0)
        frame[y:y + h, x:x + w] = blurred_face

    # Display FPS and instructions on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, instruction_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with blurred faces and FPS display
    cv2.imshow('Face Blurred', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
video_capture.release()
cv2.destroyAllWindows()
