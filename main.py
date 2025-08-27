import cv2
import streamlit as st
import numpy as np
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# Fungsi untuk membuat arsitektur model yang lebih baik
def create_model_architecture(input_shape=(28, 28, 1), num_classes=10):
    """
    Membuat arsitektur model CNN yang lebih sesuai untuk MNIST
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(28, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])
    
    # Kompilasi model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Load model dengan error handling yang lebih baik
@st.cache_resource
def load_model():
    try:
        # Method 1: Standard load
        model = keras.models.load_model('MNISTMODELCONV2D.keras')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Standard load failed: {str(e)[:100]}...")
        
        try:
            # Method 2: Load dengan compile=False
            model = keras.models.load_model('MNISTMODELCONV2D.keras', compile=False)
            model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            st.success("✅ Model loaded with compile=False!")
            return model
        except Exception as e2:
            st.warning(f"⚠️ Load with compile=False failed: {str(e2)[:100]}...")
            
            try:
                # Method 3: Load weights saja
                model = create_model_architecture()
                model.load_weights('MNISTMODELCONV2D.keras')
                st.success("✅ Weights loaded successfully!")
                return model
            except Exception as e3:
                st.error(f"❌ All loading methods failed: {str(e3)[:100]}...")
                st.info("ℹ️ Creating a new model with random weights for demonstration.")
                model = create_model_architecture()
                return model

# Load model
model = load_model()

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
        st.image('https://placehold.co/350x350/CA3433/FFFFFF?text=Akbar', 
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
