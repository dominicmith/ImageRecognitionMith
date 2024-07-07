import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('model.h5')


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    return img_array

def get_class_name(class_id):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return class_names[class_id]


def predict_image(image_path):
    
    processed_image = preprocess_image(image_path)

    
    predictions = model.predict(processed_image)

    
    class_id = np.argmax(predictions[0])
    class_name = get_class_name(class_id)

    return class_name


image_path = 'path/to/your/image.jpg'
prediction = predict_image(image_path)
print(f'Predicted class: {prediction}')
