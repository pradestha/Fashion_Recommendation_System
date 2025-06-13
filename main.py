import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit app title and layout
st.set_page_config(page_title="Fashion Recommender System", layout="wide")
st.title('👗 Fashion Recommender System')
st.markdown("Upload an image of clothing, and get recommendations for similar items!")


def save_uploaded_file(uploaded_file):
    try:
        if not uploaded_file.type.startswith('image/'):
            st.error("Please upload an image file.")
            return 0

        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error: {e}")
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# File upload section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)

        # Show recommended images
        st.subheader("Recommended Items:")
        cols = st.columns(5)

        for i, col in enumerate(cols):
            with col:
                recommended_image = Image.open(filenames[indices[0][i]])
                col.image(recommended_image, use_container_width=True, caption=f"Recommendation {i + 1}")

    else:
        st.header("Some error occurred in file upload")