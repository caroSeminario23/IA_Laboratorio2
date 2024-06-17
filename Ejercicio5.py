import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import Label

# Cargar el modelo de detección de emociones preentrenado
emotion_model = load_model('emotion_detection_model.h5')

# Etiquetas de emociones
emotion_labels = ['Enojo', 'Disgusto', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']

# Capturar video de la cámara
cap = cv2.VideoCapture(0)

# Función para capturar y mostrar el video de la cámara
def show_camera():
    ret, frame = cap.read()
    if not ret:
        return

    # Llamar a la función detect_emotion para procesar el frame actual
    detect_emotion(frame)

    # Programar la siguiente llamada a show_camera después de 10 ms
    root.after(10, show_camera)

# Función para detectar emociones en el frame actual
def detect_emotion(frame):
    # Convertir el frame a escala de grises y detectar caras
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Recortar y redimensionar la región de interés (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)  # Añadir una dimensión para el canal

        # Predecir la emoción
        preds = emotion_model.predict(roi)
        emotion_label_idx = np.argmax(preds)
        emotion_label_text = emotion_labels[emotion_label_idx]

        # Mostrar la emoción detectada y la sugerencia
        emotion_label.config(text=f"Emoción detectada: {emotion_label_text}")
        suggestion_label.config(text=suggestions.get(emotion_label_text, "No hay sugerencia"))

        # Dibujar puntos sobre los ojos, la nariz y la boca
        draw_feature_points(frame, x, y, w, h)

    # Mostrar el frame modificado con los puntos y el rectángulo
    cv2.imshow('Detected Emotions', frame)
    cv2.waitKey(1)

# Función para dibujar puntos en los ojos, la nariz y la boca
def draw_feature_points(frame, x, y, w, h):
    # Definir las coordenadas de los puntos
    eye_left = (x + int(0.3 * w), y + int(0.35 * h))
    eye_right = (x + int(0.7 * w), y + int(0.35 * h))
    nose = (x + int(0.5 * w), y + int(0.55 * h))
    mouth_left = (x + int(0.35 * w), y + int(0.7 * h))
    mouth_right = (x + int(0.65 * w), y + int(0.7 * h))

    # Dibujar los puntos como círculos sobre el frame
    cv2.circle(frame, eye_left, 2, (0, 255, 0), -1)
    cv2.circle(frame, eye_right, 2, (0, 255, 0), -1)
    cv2.circle(frame, nose, 2, (0, 255, 0), -1)
    cv2.circle(frame, mouth_left, 2, (0, 255, 0), -1)
    cv2.circle(frame, mouth_right, 2, (0, 255, 0), -1)

# Función para detectar caras en una imagen en escala de grises
def detect_faces(gray_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    return faces

# Crear la ventana de Tkinter
root = tk.Tk()
root.title("Reconocimiento de Estado de Ánimo")
root.configure(bg='#40E0D0')  # Color turquesa

# Crear etiquetas para mostrar la emoción y la sugerencia
emotion_label = Label(root, text="", font=('Helvetica', 20, 'italic'), bg='#40E0D0')
emotion_label.pack(pady=20)

suggestion_label = Label(root, text="", font=('Helvetica', 16, 'italic'), bg='#40E0D0')
suggestion_label.pack(pady=20)

# Sugerencias para mejorar el estado de ánimo
suggestions = {
    'Tristeza': "Recuerda que no estás solo. Siempre hay una salida.",
    'Enojo': "Respira profundamente y busca la calma.",
    'Miedo': "Todo estará bien. Tómate un momento para relajarte.",
    'Disgusto': "Intenta enfocarte en algo positivo.",
    'Sorpresa': "Es un buen momento para descubrir algo nuevo.",
    'Neutral': "Todo está en equilibrio. Sigue así.",
    'Felicidad': "Sigue disfrutando de este momento."
}

# Iniciar la función para mostrar el video de la cámara
show_camera()

# Ejecutar la ventana de Tkinter
root.mainloop()

# Liberar la cámara
cap.release()
cv2.destroyAllWindows()
