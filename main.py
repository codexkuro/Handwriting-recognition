import cv2
import streamlit as st
import numpy as np
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

try:
    # Coba load model yang sudah ada
    model = keras.models.load_model('MNISTMODELCONV2D.keras')
except:
    # Jika gagal, gunakan model pre-trained dari TensorFlow
    try:
        model = keras.models.load_model('path/to/pretrained/model.h5')
    except:
        # Fallback: Gunakan model built-in Keras
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Preprocess data
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # Build simple model
        model = keras.Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train cepat dengan sedikit data
        model.fit(x_train[:1000], y_train[:1000], epochs=1, verbose=0)

def main():
    st.markdown("""
        <style>
            .header-container {
                background: linear-gradient(135deg, #CA3433, #FF6B6B);
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                margin-bottom: 2rem;
                text-align: center;
            }
            
            .header-container1 {
                background: linear-gradient(135deg, #CA3433, #FF6B6B);
                padding: 1.5rem;
                border-radius: 1rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                margin-bottom: 2rem;
                text-align: center;
            }

            .header-title {
                color: white;
                font-size: 36px;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }

            .header-subtitle {
                color: #F8F9FA;
                font-size: 18px;
            }
            
            .prediction-box {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 5px solid #28a745;
                margin-top: 1rem;
            }
            
            @media (max-width: 768px) {
                .header-title {
                    font-size: 28px;
                }
                .header-subtitle {
                    font-size: 16px;
                }
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="header-container">
            <div class="header-title">Handwriting Recognition</div>
            <div class="header-subtitle">Exercise deep learning project made by Codexkuro/Akbar</div>
        </div>
    """, unsafe_allow_html=True)
    
    col, col1, col2 = st.columns([1, 2, 1])
    with col1:
        st.write("### ✍️ Draw a digit (0-9)")
        
        # Canvas untuk gambar
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("🎯 Predict", type='primary', use_container_width=True):
            if canvas_result.image_data is not None:
                with st.spinner('Predicting...'):
                    pred_class, confidence = model_prediction(canvas_result)
                    
                    # Tampilkan hasil dengan style yang bagus
                    st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Prediction Result</h3>
                            <p><strong>Digit:</strong> <span style="font-size: 24px; color: #CA3433;">{pred_class}</span></p>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Tampilkan gambar yang diproses
                    st.write("Processed image (28x28 pixels):")
                    processed_img = preprocess_image(canvas_result)
                    st.image(processed_img, width=100)
            else:
                st.warning("⚠️ Please draw a digit on the canvas first!")

    st.markdown("""
        <div class="header-container1">
            <div class="header-subtitle">👨‍💻 About the Developer</div>
        </div>
    """, unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns([1, 2, 1])
    with col4:
        # Placeholder untuk foto - ganti dengan path foto Anda
        st.image('Akbar.jpg', 
                 caption='Programmer Sigma', width=350)
    
    st.markdown("""
    Halo, saya Akbar. Ini adalah website exercise deep learning saya untuk menguji model yang sudah dilatih. 
    Di sini Anda dapat menggambar angka 0-9 pada kanvas, dan model neural network akan memprediksi angka yang Anda tulis.
    
    **Note:** Jika model tidak dapat memuat weights asli, aplikasi akan menggunakan model dengan weights acak untuk demonstrasi.
    """)

def preprocess_image(input_image):
    # Ambil hasil canvas
    img = input_image.image_data.astype(np.uint8)

    # Buang alpha channel (RGBA → RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Ubah jadi grayscale
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # Invert colors (background hitam, digit putih → background putih, digit hitam)
    gray_img = cv2.bitwise_not(gray_img)

    # Resize ke 28x28
    resized = cv2.resize(gray_img, (28, 28))

    # Normalisasi
    resized = resized.astype('float32') / 255.0

    return resized

def model_prediction(input_image):
    # Preprocess gambar
    processed_img = preprocess_image(input_image)
    
    # Untuk ditampilkan
    display_img = (processed_img * 255).astype(np.uint8)
    
    # Tambahkan channel dimension + batch dimension
    img_for_prediction = np.expand_dims(processed_img, axis=-1)  # (28,28) → (28,28,1)
    img_for_prediction = np.expand_dims(img_for_prediction, axis=0)   # (28,28,1) → (1,28,28,1)

    # Prediksi
    pred = model.predict(img_for_prediction, verbose=0)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)
    
    return predicted_class, confidence

if __name__ == "__main__":
    main()
