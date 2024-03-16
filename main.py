import cv2

haar_cascade = 'cars.xml'
video = 'car2.mp4'

cap = cv2.VideoCapture(video)
car_cascade = cv2.CascadeClassifier(haar_cascade)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

while True:
    # Read frames from the video
    ret, frames = cap.read()

    # Check if frames were successfully read
    if not ret:
        print("Error: Unable to read frames from video.")
        break

    # Convert frames to grayscale
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # To draw a rectangle around each car
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display frames in a window
    cv2.imshow('video', frames)

    # Check for the 'Esc' key press to exit
    if cv2.waitKey(33) == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
