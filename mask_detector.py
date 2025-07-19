import cv2

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start your webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles and simulate mask detection
    for (x, y, w, h) in faces:
        # Draw rectangle on face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Simulated condition (you can randomize or add logic)
        label = "Mask Detected"  # Replace this with prediction later
        color = (0, 255, 0)  # Green for mask

        # Display label
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the result
    cv2.imshow('Face Mask Detector (Simulated)', frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and destroy windows
cap.release()
cv2.destroyAllWindows()
