# Import libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from urllib.request import urlopen

# Load the trained model
model = tf.keras.models.load_model('model/trash.h5')

# Create a function to preprocess the image and make predictions
def predict_trash(img):
    img = img.resize((180, 180))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    result = model.predict(np.expand_dims(img, axis=0))

    class_id = np.argmax(result)

    classes = {
        0: 'Cardboard',
        1: 'Glass',
        2: 'Metal',
        3: 'Paper',
        4: 'Plastic',
        5: 'Trash'
    }

    prediction = classes[class_id]

    return prediction, result[0][class_id]

def main():
    st.title("Trash Type Detector")

    activities = ["Upload Image", "Image Link", "About", "Contact"]
    choice = st.sidebar.selectbox("Pages", activities)

    if choice == 'Upload Image':
        st.subheader("Trash Detector via Local Image Upload")
        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            prediction, confidence = predict_trash(img)
            st.write(f'This is an image of a **{prediction}** (Confidence: {confidence:.2%})')
            st.image(img, caption="Uploaded Image", use_column_width=True)

    if choice == 'Image Link':
        st.subheader("Trash Detector via Image Link")
        raw_url = st.text_input("Please enter an image link", "https://example.com")
        if st.button("Detect"):
            try:
                image = Image.open(urlopen(raw_url))
                prediction, confidence = predict_trash(image)
                st.write(f'This is an image of a **{prediction}** (Confidence: {confidence:.2%})')
                st.image(image, caption="Image from URL", use_column_width=True)

            except Exception as e:
                st.write("Error loading image from the provided link.")

    if choice == 'About':
        st.subheader("What is Trash Type Detector?")
        st.info("This application is designed for a student project using convolutional neural network-based modeling, with a model accuracy of approximately 66%. The model has been trained on approximately 2527 different images.")

    if choice == 'Contact':
        st.subheader("Contact us:")
        st.info("For further information, feel free to contact us at f.mirfaizi@gmail.com")

if __name__ == '__main__':
    main()


# Run : python -m streamlit run web_app.py --server.port 8080