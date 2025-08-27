import cv2
import streamlit as st
import numpy as np
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas


# load model
model = keras.models.load_model('MNISTV1.keras', compile=False)
def create_model_architecture(input_shape=(28, 28, 1), num_classes=10):
    """
    Membuat arsitektur model CNN sederhana untuk MNIST
    """
    model = keras.Sequential([
        keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        keras.layers.Conv2D(28, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])
    
    return model

# Coba approach yang berbeda untuk load model
try:
    # Method 1: Standard load
    model = keras.models.load_model('MNISTMODELCONV2D.keras')
except:
    try:
        # Method 2: Load dengan custom objects (jika ada)
        model = keras.models.load_model('MNISTMODELCONV2D.keras', compile=True)
    except:
        try:
            # Method 3: Load weights saja
            model = keras.models.load_model('MNISTMODELCONV2D.keras', compile=False)
            # Compile manual setelah load
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            print(f"Error: {e}")
            # Fallback: Bangun model manual dan load weights
            model = create_model_architecture()  # Anda perlu definisikan arsitektur model
            model.load_weights('MNISTMODELCONV2D.keras')

def main():
    st.markdown("""
        <style>
            .header-container {
                background-color: #CA3433;
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                margin-bottom: 2rem;
            }
            
            .header-container1 {
                background-color: #CA3433;
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                margin-bottom: 2rem;
            }

            .header-title {
                color: white;
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 0.5rem;
            }

            .header-subtitle {
                color: #CCCCCC;
                font-size: 28px;
                text-align: center;
            }
            
            @media (max-width: 768px) {
                .header-title {
                    font-size: 28px;
                }
                .header-subtitle {
                    font-size: 14px;
                }
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
            <div class="header-container">
                <div class="header-title">Handwritting Recognition</div>
                <div class="header-subtitle">Exercise deep learning project made by Codexkuro/Akbar</div>
            </div>
        """, unsafe_allow_html=True)
    col, col1, col2 = st.columns([1, 2, 1])
    with col1:
        # canvas untuk gambar
        gambar = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=8,
            stroke_color="black",
            background_color="white",
            width=400,
            height=300,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("Prediksi", type='primary', use_container_width=True):
            if gambar.image_data is not None:
                pred_class = model_prediction(gambar)
                st.success(f"Hasil prediksi: {pred_class}")
            else:
                st.warning("Silakan gambar dulu di kanvas!")

    st.markdown("""
                    <div class="header-container1">
                        <div class="header-subtitle">👨‍💻 Tentang Pengembang</div>
                    </div>
                    """, unsafe_allow_html=True)
    col3, col4, col5 = st.columns([1,2,1])
    with col4:
        st.image('Akbar.jpg', caption='Programmer sigma', width=350)
    st.markdown("Halo, gw akbar. Ini adalah website exercise deep learning gw untuk mengetest model yang sudah gw train. Jadi disini tuh methodnya kalian gambar angka 1-9 di kanvas, nanti model neural network simpel bakal memprediksi angka yang kalian tulis dan menghasilkan output 1-9 juga. Sorry kalo misalnya outputnya rads ngaco, soalnya memang website ini dibuat sekadar untuk testing saja hehe.")

def model_prediction(input_image):
    # ambil hasil canvas
    img = input_image.image_data.astype(np.uint8)

    # buang alpha channel (RGBA → RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # ubah jadi grayscale
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    gray_img = cv2.bitwise_not(gray_img)

    # resize ke 28x28
    resized = cv2.resize(gray_img, (28, 28))

    # normalisasi
    resized = resized / 255

    # tambahkan channel dimension + batch dimension
    resized = np.expand_dims(resized, axis=-1)  # (28,28) → (28,28,1)
    resized = np.expand_dims(resized, axis=0)   # (28,28,1) → (1,28,28,1)

    # prediksi
    pred = model.predict(resized)
    return np.argmax(pred)

if __name__ == "__main__":
    main()
