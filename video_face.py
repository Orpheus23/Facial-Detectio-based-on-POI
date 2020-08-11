
import cv2
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras import models
from keras.layers.core import Lambda, Flatten
from keras_vggface.vggface import VGGFace
vggface = VGGFace(model='senet50')
from mtcnn import MTCNN
K.set_image_data_format('channels_last')


def readimg(path, detector=MTCNN()):
    img = tf.image.decode_jpeg(tf.io.read_file(path))
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)
    face = detector.detect_faces(img)
    for i in face:
        (x, y, w, h) = i['box']
        img2 = img[y:y+h, x:x+w, :]
    try:
        img = cv2.resize(img2, (250, 250))
    except:
        img = cv2.resize(img, (250, 250))
    return img


def cos_distance(y):
    y_true, y_pred=y[0],y[1]
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(K.sum((y_true * y_pred), axis=-1))


def get_siamese_model(input_shape):
    # Input tensors

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = models.Sequential()
    model2 = VGGFace(include_top=True, model='senet50', weights='vggface', input_shape=input_shape)

    last_layer = model2.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    model.add(Model(model2.input, x))

    for layer in model.layers:
        layer.trainable = False

    # cos_distance = merge([vec_a, vec_b], mode='cos', dot_axes=1)
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    # output_cos = tf.keras.layers.Dot(axes=-1)([encoded_l,encoded_r])
    dist_output = Lambda(cos_distance)([encoded_l, encoded_r])

    '''
    L1_layer = Lambda(lambda tensors:tf.keras.metrics.AUC().update_state(tensors[0],tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    '''

    siamese_net = Model(inputs=[left_input, right_input], outputs=dist_output)

    return siamese_net


def predict(threshold, output):
    if output >= threshold:
        return output, "same person"
    elif output <= (1-threshold):
        return output, "different person"
    else:
        return output, "Not conclusive"


def detect(detector, k, webcam, model, threshold, vidpath = None):
    if webcam:
        video_cap = cv2.VideoCapture(1)  # use 0,1,2..depanding on your webcam
    else:
        video_cap = cv2.VideoCapture(vidpath)
    while True:
        # Capture frame-by-frame
        ret, img = video_cap.read()

        # converting to gray image for faster video processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input1 = [np.zeros([1, 250, 250, 3]) for i in range(2)]

        rects = detector.detect_faces(img)
        # if at least 1 face detected
        if len(rects) >= 0:
            # Draw a rectangle around the faces
            for i in rects:
                (x, y, w, h) = i['box']
                img2 = img[y:y + h, x:x + w, :]
                try:
                    img3 = cv2.resize(img2, (250, 250))
                    input1[0][0, :, :, :] = img3
                    input1[1][0, :, :, :] = k
                    output = model.predict(input1)
                    if predict(threshold, output)[1] == "same person":
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                except:
                    print("error")
            cv2.imshow('Face Detection on Video', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()


def main():
    model = get_siamese_model((250, 250, 3))
    model.summary()
    detector = MTCNN()
    POI = Path1
    k = readimg(POI, detector)
    threshold = 0.6
    webcam = False

    detect(detector, k, webcam, model, threshold)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()