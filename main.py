import cv2
import numpy as np
import time
import winsound
from tensorflow.keras.models import load_model

# Load model
model = load_model("drowsiness_model.h5")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

# Time + frame logic
closed_start_time = None
closed_frames = 0

DROWSY_THRESHOLD = 5      # seconds
FRAME_THRESHOLD = 15      # frames

# Beep control
last_beep_time = 0
BEEP_INTERVAL = 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        closed_frames = 0
        closed_start_time = None

    else:
        # ✅ FIX: Select only largest face (closest to camera)
        largest_face = max(faces, key=lambda rect: rect[2]*rect[3])
        (x, y, w, h) = largest_face

        eyes_closed = True  # same logic

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        face_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # If no eyes detected → assume closed (your original logic)
        if len(eyes) == 0:
            eyes_closed = True

        for (ex, ey, ew, eh) in eyes:
            eye = face_gray[ey:ey+eh, ex:ex+ew]

            # Preprocess
            eye_img = cv2.resize(eye, (64, 64))
            eye_img = eye_img / 255.0
            eye_img = np.reshape(eye_img, (1, 64, 64, 1))

            prediction = model.predict(eye_img, verbose=0)[0][0]

            # Same threshold
            if prediction >= 0.6:
                eyes_closed = False
                label = "Eyes Open 🙂"
            else:
                label = "Eyes Closed 😴"

            # Draw eye box
            cv2.rectangle(frame, (x+ex, y+ey),
                          (x+ex+ew, y+ey+eh), (0,255,0), 2)

            cv2.putText(frame, label, (x+ex, y+ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 🔥 SAME FRAME LOGIC
        if eyes_closed:
            closed_frames += 1
        else:
            closed_frames = 0
            closed_start_time = None

        # 🔥 SAME TIME LOGIC
        if closed_frames > FRAME_THRESHOLD:
            if closed_start_time is None:
                closed_start_time = time.time()

            elapsed = time.time() - closed_start_time

            if elapsed > DROWSY_THRESHOLD:
                cv2.putText(frame, "🚨 DRIVER IS DROWSY 🚨", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                current_time = time.time()
                if current_time - last_beep_time > BEEP_INTERVAL:
                    winsound.Beep(1000, 500)
                    winsound.Beep(1000, 500)
                    last_beep_time = current_time
        else:
            cv2.putText(frame, "Driver is Alert 🙂", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
