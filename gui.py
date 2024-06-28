import cv2
import numpy as np
from keras.api.models import model_from_json

json_file = open("combined_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("context_model_weights.weights.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)



def extract_features(frames):
    features = []
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if len(face) == 1:
            (p, q, r, s) = face[0]
            face_img = frame_gray[q:q+s, p:p+r]
            face_img = cv2.resize(face_img, (48, 48))
            features.append(face_img)
        else:
            features.append(np.zeros((48, 48), dtype=np.uint8))
    if len(features) != 5:
        print("Error: Detected less than 5 faces in the sequence.")
        return None
    features = np.array(features)
    features = features.reshape(1, 5, 48, 48, 1) 
    features = features / 255.0  
    return features


labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


webcam = cv2.VideoCapture(0)



while True:
    frames = []
    for _ in range(5):  
        _, frame = webcam.read()
        frames.append(frame)

    features = extract_features(frames)

    
    if features.size != 0:
        prediction = model.predict(features)
        prediction_label = labels[np.argmax(prediction)]

        
        cv2.putText(frames[-1], prediction_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Cultural Contextual Awareness", frames[-1])

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

