
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Ruta al modelo
modelo_path = "modelo_emociones90.h5"
model = load_model(modelo_path)

# Clases del modelo
clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("📸 Presiona la tecla 'c' para capturar una foto y predecirla")
print("❌ Presiona 'q' para salir sin capturar")

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    cv2.imshow("Presiona 'c' para capturar", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        captured = True
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Si se capturó la imagen, hacer predicción
if captured:
    print("✅ Imagen capturada. Realizando inferencia...")

    # Preprocesar imagen
    img = cv2.resize(frame, (64, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = model.predict(img)[0]
    clase_idx = np.argmax(pred)
    confianza = pred[clase_idx]
    clase = clases[clase_idx]

    # Mostrar resultado
    print(f"🎯 Predicción: {clase} ({confianza:.2f})")

    # Mostrar imagen con resultado
    cv2.putText(frame, f"{clase} ({confianza:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Resultado", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
