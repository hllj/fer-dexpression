import cv2
import numpy
import tensorflow as tf
from keras.models import load_model

EMOTION_DICT = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


def make_prediction(model, path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_clip = img[y:y + h, x:x + w]
        img = cv2.resize(face_clip, (350, 350))
    cv2.imshow("Image", img)
    # read the processed image then make prediction and display the result
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    read_image_final = img / 255.0  # normalizing the image

    top_pred = model.predict(read_image_final)
    emotion_label = top_pred[0].argmax()
    print("Predicted Expression Probabilities")
    for idx, emotion in enumerate(EMOTION_DICT):
        print(emotion, " : ", top_pred[0][idx])
    print("Dominant Probability = " + str(EMOTION_DICT[emotion_label]) + ": " + str(max(top_pred[0])))

model = load_model('model.h5')
make_prediction(model, 'examples/test.jpg')
make_prediction(model, 'examples/test_happy.jpg')
make_prediction(model, 'examples/test_angry.jpg')
make_prediction(model, 'examples/test_surprise.jpg')

k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
