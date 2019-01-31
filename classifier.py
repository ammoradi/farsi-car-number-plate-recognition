from keras.models import Sequential
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.core import Flatten, Dense
from HodaDatasetReader import read_hoda_dataset, __resize_image
import numpy as np
import cv2 as cv


def cnn_classify(image):
    # initialize the model
    print("Compiling model...")
    model = Sequential()
    inputShape = (32, 32, 1)

    # Convolution Layer 1
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
    # SubSampling 1
    model.add(AveragePooling2D())

    # Convolution Layer 2
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    # SubSampling 2
    model.add(AveragePooling2D())

    model.add(Flatten())

    # Connected Layer 1
    model.add(Dense(units=120, activation='relu'))

    # Connected Layer 2
    model.add(Dense(units=84, activation='relu'))

    # Connected Layer 3
    model.add(Dense(units=10, activation='softmax'))

    # uncomment below line for using "pre-trained" model's weights.
    model.load_weights('cnn_model/weights.hdf5')

    test_data = np.resize(image, (1, 32, 32, 1))

    probs = model.predict(test_data)
    prediction = probs.argmax(axis=1)

    print(probs)

    # show the image and prediction
    print("Predicted: {},".format(prediction[0]))
    cv.imshow("Digit", image)
    cv.waitKey(0)

    return prediction


def knn_classify(image):
    opencv_knn = cv.ml.KNearest_create()
    train_images, train_labels = read_hoda_dataset(dataset_path='./digits/Test 20000.cdb',
                                                 images_height=32,
                                                 images_width=32,
                                                 one_hot=False,
                                                 reshape=True)

    test_data = np.zeros(shape=[1, 32, 32], dtype=np.float32)

    img = image
    # img = __resize_image(src_image=img, dst_image_height=32, dst_image_width=32)
    img = img / 255
    img = np.where(image >= 0.5, 1, 0)
    test_data[0] = img

    test_data = test_data.reshape(-1, 32 * 32)

    opencv_knn.train(train_images, cv.ml.ROW_SAMPLE, train_labels)
    ret, results, neighbours, dist = opencv_knn.findNearest(test_data, 3)

    result = np.reshape(results, -1)
    return result


