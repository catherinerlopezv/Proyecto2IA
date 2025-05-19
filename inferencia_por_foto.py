import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import defaultdict
import os

# =======================
# CONFIGURACIÓN
# =======================

modelo_path = "modelo_emociones90.h5"
model = load_model(modelo_path)

clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

colores = {
    'angry': (0, 0, 255),       # rojo
    'disgust': (0, 128, 0),     # verde oscuro
    'fear': (128, 0, 128),      # púrpura
    'happy': (255, 255, 0),     # amarillo
    'neutral': (200, 200, 200), # gris claro
    'sad': (0, 0, 200),         # azul
    'surprise': (0, 255, 255)   # celeste
}

# =======================
# CAPTURA DE FOTO
# =======================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("📸 Presiona la tecla 'c' para capturar una foto y predecirla")
print("❌ Presiona 'q' para salir sin capturar")

captured = False
frame = None

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

# =======================
# INFERENCIA
# =======================

if captured:
    print("✅ Imagen capturada. Realizando inferencia...")

    # Preprocesamiento
    img = cv2.resize(frame, (64, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    clase_idx = np.argmax(pred)
    clase = clases[clase_idx]
    confianza = pred[clase_idx]

    # Guardar inferencia en archivo
    with open("registro_inferencias.txt", "a") as f:
        f.write(clase + "\n")

    # Contador de emociones
    conteo = defaultdict(int)
    if os.path.exists("registro_inferencias.txt"):
        with open("registro_inferencias.txt", "r") as f:
            for linea in f:
                linea = linea.strip().lower()
                if linea in clases:
                    conteo[linea] += 1

    # =======================
# VISUALIZACIÓN COMPACTA
# =======================
result_frame = frame.copy()
result_frame = cv2.resize(result_frame, (900, 500))  # ancho fijo

# Crear canvas negro debajo de la imagen
info_height = 320
canvas = np.zeros((500 + info_height, 900, 3), dtype=np.uint8)
canvas[:, :] = (20, 20, 20)  # fondo gris oscuro

canvas[0:500, :] = result_frame

# Resultado principal
cv2.putText(canvas, f"Resultado: {clase.upper()} ({confianza:.2f})", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, colores.get(clase, (255, 255, 255)), 3)

# Barras de predicción por emoción
for i, c in enumerate(clases):
    barra_y = 70 + i * 30
    porcentaje = pred[i]
    longitud = int(200 * porcentaje)

    texto = f"{c.upper()} ({porcentaje:.2f})"
    color = colores.get(c, (255, 255, 255))

    cv2.putText(canvas, texto, (30, 500 + barra_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    cv2.rectangle(canvas, (250, 500 + barra_y - 10), (250 + longitud, 500 + barra_y + 5), color, -1)
    cv2.rectangle(canvas, (250, 500 + barra_y - 10), (250 + 200, 500 + barra_y + 5), (255, 255, 255), 1)

# Conteo total por emoción
cv2.putText(canvas, "Total por emocion:", (500, 500 + 40),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

for i, c in enumerate(clases):
    texto = f"{c.capitalize():<8}: {conteo.get(c, 0)}"
    color = colores.get(c, (255, 255, 255))
    cv2.putText(canvas, texto, (500, 500 + 70 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

# Mostrar todo
cv2.imshow("Resultado", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
