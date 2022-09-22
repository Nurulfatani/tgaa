import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps
import time

def our_image_classifier(image):
    '''
            Function that takes the path of the image as input and returns the closest predicted label as output
            '''
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = tensorflow.keras.models.load_model(
        'songket.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (
        image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    labels = {0: "Bungog Delima", 1: "Bungong Meulu 1", 2 : "Bungong Meulu 2", 3 :"Bungong Meurante", 4 :"Pinto Aceh", 5 :"Pucok Mueria", 6 :"Tidak dikenali"}
    predictions = model.predict(data).tolist()
    best_outcome = predictions[0].index(max(predictions[0]))
    print(labels[best_outcome])
    return labels[best_outcome]
