import cv2
import numpy as np
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit the application.")

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale (for better processing)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try facial expression analysis using DeepFace
    try:
        # Analyze frame for emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion
        dominant_emotion = analysis[0]['dominant_emotion']

        # Display the emotion on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        # Handle errors (e.g., no face detected)
        print(f"Error: {e}")
        cv2.putText(frame, 'No face detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the video feed with detected emotion
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()