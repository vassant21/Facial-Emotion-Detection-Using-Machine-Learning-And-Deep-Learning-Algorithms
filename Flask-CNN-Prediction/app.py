import cv2
from flask import Flask, render_template, Response
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)
# Function to load the model and weights
def load_model_and_weights():
    try:
        # Loading the saved model
        model_path = "C:/Users/User/Desktop/22CP05-PROJECT/Face/Prediction/250efacemodel.json"
        weight_path = "C:/Users/User/Desktop/22CP05-PROJECT/Face/Prediction/250efacemodel.h5"
        
        model = model_from_json(open(model_path, "r").read())
        model.load_weights(weight_path)

        return model
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return None

# Loading the model and weights
model = load_model_and_weights()
# haarcascade_frontalface_default
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to process frames and perform emotion detection
def detect_emotions():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        res, frame = cap.read()
        height, width, channel = frame.shape
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
                roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)[0]
                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

                # Find the index of the emotion with the highest probability
                max_index = np.argmax(predictions)

                # Display probabilities for each emotion
                textY = y + h + 20  # Adjust the vertical position of the text below the detected face
                for i, emotion_prob in enumerate(predictions):
                    emotion_label = f"{emotions[i]}: {emotion_prob*100:.2f}%"
                    color = (0, 255, 0) if i == max_index else (255, 255, 255)
                    cv2.putText(frame, emotion_label, (x, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    textY += 20  # Increase the vertical offset for the next emotion label
        except:
            pass

        # Convert the processed frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotions(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
