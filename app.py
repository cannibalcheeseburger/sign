#Import necessary libraries
from flask import Flask, render_template, Response
import cv2 as cv
import csv
import mediapipe as mp
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark


#Initialize the Flask app
app = Flask(__name__)
camera = cv.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
)

keypoint_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = get_result(frame)
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def get_result(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

            debug_image = draw_landmarks(debug_image, landmark_list)

            debug_image = draw_info_text(
                debug_image,
                handedness,
                keypoint_classifier_labels[hand_sign_id])



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)