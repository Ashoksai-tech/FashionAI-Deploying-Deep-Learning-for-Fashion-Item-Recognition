import streamlit as st
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from PIL import Image
import numpy as np
import os


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/fashion_trained_mnist_model.h5"
model = tf.keras.models.load_model(model_path)

# import os

# # Determine the working directory
# working_dir = os.path.dirname(os.path.abspath(__file__))

# # Define the model path
# model_path = os.path.join(working_dir, 'trained_model', 'fashion_mnist_model.h5')

# # Check if the model file exists
# if not os.path.isfile(model_path):
#     raise FileNotFoundError(f"No file found at {model_path}")

# # Load the model
# model = tf.keras.models.load_model(model_path)


# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#function to preprocess the uploaded image
def preprocess_image(image):
    img=Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L') #convert to gray scale
    img_array = np.array(img)/255
    img_array = img_array.reshape((1,28,28,1))
    return img_array

#streamlit app
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("uploaded an image",
                                  type=['jpg','jpeg','png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1,col2 = st.columns(2)
    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            # st.write(str(result))
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')

