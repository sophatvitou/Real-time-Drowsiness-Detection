import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# ===============================
# Load model
# ===============================
model = load_model("drowsiness_eye_model.h5")
print("Model loaded")

# ===============================
# Load Haar cascades
# ===============================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    "haarcascade_eye.xml"
)

# ===============================
# Preprocess eye image
# MUST match training
# ===============================
def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, (224, 224))
    eye_img = eye_img / 255.0
    eye_img = np.expand_dims(eye_img, axis=-1)  # (224,224,1)
    eye_img = np.expand_dims(eye_img, axis=0)   # (1,224,224,1)
    return eye_img

# ===============================
# Camera
# ===============================
cap = cv2.VideoCapture(0)

closed_count = 0
DROWSY_THRESHOLD = 20  # frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "NO FACE"

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_input = preprocess_eye(eye_img)

            prediction = model.predict(eye_input, verbose=0)[0][0]

            if prediction < 0.5:
                closed_count += 1
                status = "EYES CLOSED"
            else:
                closed_count = 0
                status = "EYES OPEN"

            cv2.rectangle(
                frame,
                (x+ex, y+ey),
                (x+ex+ew, y+ey+eh),
                (0,255,0),
                2
            )
            break

    # ===============================
    # Drowsiness alert
    # ===============================
    if closed_count >= DROWSY_THRESHOLD:
        cv2.putText(
            frame, "DROWSY ALERT!",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0,0,255), 3
        )

    cv2.putText(
        frame, status,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0,255,0), 2
    )

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
