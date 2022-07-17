from lzma import MODE_FAST
import numpy as np
import cv2
import tensorflow as tf

"""# define constant"""

classes = 101 # Tu 0 den 100 tuoi
output_indexes = np.array([i for i in range(0, 101)])
target_size = (224, 224)
shape = (224, 224, 3)
# model_path = "model_class" #***
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
model_path="C:\\Users\\anhtr\\OneDrive\\Máy tính\\python\\API\\model_class.h5"
"""# Tồn tại khuôn mặt"""


model = tf.keras.models.load_model(model_path)
def response(frame):
    """# Load model"""
    """# Dự đoán tuổi của ảnh upload
    # Plot frame, rectangle
    """
    res, im_jpg = cv2.imencode(".jpg", frame)
    try:
      faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
      for (x, y, w, h) in faces:
        if w > 130:  # Bo qua cac mat nho
            # Ve hinh chu nhat quanh mat
            tt = cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)  # draw rectangle to main image
            #cv2_imshow(tt)
            # Crop mat
            detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            try:
                # Them magin
                margin = 30
                margin_x = int((w * margin) / 100);
                margin_y = int((h * margin) / 100)
                detected_face = frame[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w + margin_x)]
            except:
              print('detected face has no margin')
            # Age Prediction
            try:
              def predict_age(detected_face):
                detected_face = detected_face/255.0 # to normal scale
                input = cv2.resize(detected_face, target_size) # resize from cv2
                input = input.reshape(1, *shape) # reshape to standard shape
                output = model.predict(input)
                #print(output)
                return int(np.floor(np.sum(output * output_indexes, axis=1))[0])

              predicted_age = str(min(predict_age(frame), predict_age(detected_face)))
              print(predicted_age)

              # Ve khung thong tin
              info_box_color = (46, 200, 255)
              triangle_cnt = np.array(
                  [(x + int(w / 2), y), (x + int(w / 2) - 20, y - 20), (x + int(w / 2) + 20, y - 20)])
              tt = cv2.drawContours(frame, [triangle_cnt], 0, info_box_color, -1)
              #cv2_imshow(tt2)
              tt = cv2.rectangle(frame, (x + int(w / 2) - 50, y - 20), (x + int(w / 2) + 50, y - 90), info_box_color,
                            cv2.FILLED)

              tt = cv2.putText(frame, predicted_age, (x + int(w / 2.2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

              res, im_jpg = cv2.imencode(".jpg", tt)

            except Exception as e:
              print("exception", str(e))
    except Exception as e:
      print(e)

    return im_jpg

