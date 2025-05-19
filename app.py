from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
import queue

app = Flask(__name__)

# Función para obtener emoji según la emoción
def get_emoji(emocion):
    emojis = {
        'happy': '😊',
        'neutral': '😐',
        'sad': '😢',
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'surprise': '😲'
    }
    return emojis.get(emocion, '❓')

# Función para traducir emoción al español
def traducir_emocion(emocion):
    traducciones = {
        'happy': 'Feliz',
        'neutral': 'Neutral',
        'sad': 'Triste',
        'angry': 'Enojado',
        'disgust': 'Disgusto',
        'fear': 'Miedo',
        'surprise': 'Sorpresa'
    }
    return traducciones.get(emocion, 'Desconocido')

# Cargar modelo
print("Cargando modelo...")
modelo = load_model("C:/Users/infobsolorzano/Downloads/Proyecto2IA/Proyecto2IA/modelo_emociones90.h5")
print(modelo.input_shape)
print("Modelo cargado correctamente")

# Clases de emociones
clases_emociones = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Inicializar cámara
def inicializar_camara():
    camara = cv2.VideoCapture(0)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camara.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return None
    return camara

# Detector de rostros
detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Variables globales
estadisticas = {
    "personas_detectadas": 0,
    "emocion_predominante": "---",
    "conteo_emociones": {}
}
lock = threading.Lock()

frame_queue = queue.Queue(maxsize=10)
ultima_actualizacion = time.time()
debe_procesar = True

# Procesar frames
def procesar_frames():
    global estadisticas, debe_procesar, ultima_actualizacion

    cap = inicializar_camara()
    if cap is None:
        return

    while debe_procesar:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame, reintentando...")
            cap.release()
            time.sleep(1)
            cap = inicializar_camara()
            if cap is None:
                continue
            continue

        tiempo_actual = time.time()
        if tiempo_actual - ultima_actualizacion < 0.1:
            try:
                if not frame_queue.full():
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_queue.put(buffer)
            except Exception as e:
                print(f"Error al codificar frame: {e}")
            continue

        ultima_actualizacion = tiempo_actual

        try:
            frame_procesado = frame.copy()
            gray = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2GRAY)
            rostros = detector_rostros.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            emociones_detectadas = []

            for (x, y, w, h) in rostros:
                try:
                    if y < 0 or y + h > frame_procesado.shape[0] or x < 0 or x + w > frame_procesado.shape[1]:
                        continue

                    rostro = gray[y:y + h, x:x + w]
                    if rostro.size == 0:
                        continue

                    rostro_redimensionado = cv2.resize(rostro, (64, 64))  # Redimensionar a 48x48
                    rostro_rgb = cv2.cvtColor(rostro_redimensionado, cv2.COLOR_GRAY2RGB)  # Convertir a RGB
                    rostro_array = img_to_array(rostro_rgb) / 255.0  # Normalizar entre 0 y 1
                    rostro_array = np.expand_dims(rostro_array, axis=0)  # Agregar dimensión para batch
                    print(rostro_array.shape) 

                    pred = modelo.predict(rostro_array, verbose=0)[0]
                    clase_idx = np.argmax(pred)
                    emocion = clases_emociones[clase_idx]
                    emoji = get_emoji(emocion)
                    emociones_detectadas.append(emocion)

                    cv2.rectangle(frame_procesado, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    etiqueta = f"{emoji} {emocion}"
                    text_size = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame_procesado, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 0, 0), -1)
                    cv2.putText(frame_procesado, etiqueta, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                except Exception as e:
                    print(f"Error al procesar rostro: {e}")
                    continue

            conteo_actual = {}
            for emo in emociones_detectadas:
                conteo_actual[emo] = conteo_actual.get(emo, 0) + 1

            # Actualizar estadísticas globales
            with lock:
                estadisticas["personas_detectadas"] = len(rostros)
                
                # Actualizar conteo de emociones en el frame actual
                for emo in emociones_detectadas:
                    estadisticas["conteo_emociones"][emo] = estadisticas["conteo_emociones"].get(emo, 0) + 1

                # Determinar la emoción predominante
                if emociones_detectadas:
                    estadisticas["emocion_predominante"] = max(estadisticas["conteo_emociones"], key=estadisticas["conteo_emociones"].get)
                else:
                    estadisticas["emocion_predominante"] = "---"

            try:
                with lock:
                    x_offset = 10
                    y_offset = 30
                    line_spacing = 30
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    color = (0, 0, 0)
                    background_color = (255, 255, 255)

                    overlay = frame_procesado.copy()
                    cv2.rectangle(overlay, (5, 5), (220, 260), background_color, -1)
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, frame_procesado, 1 - alpha, 0, frame_procesado)

                    emocion_predom = estadisticas["emocion_predominante"]
                    texto_predom = f"Predominante: {get_emoji(emocion_predom)} {traducir_emocion(emocion_predom)}"
                    cv2.putText(frame_procesado, texto_predom, (x_offset, y_offset), font, font_scale + 0.1, (0, 0, 255), 2)

                    y_offset += 40
                    for emocion in clases_emociones:
                        contador = estadisticas["conteo_emociones"].get(emocion, 0)
                        emoji = get_emoji(emocion)
                        texto = f"{emoji} {traducir_emocion(emocion)}: {contador}"
                        cv2.putText(frame_procesado, texto, (x_offset, y_offset), font, font_scale, color, 1)
                        y_offset += line_spacing
            except Exception as e:
                print(f"Error al dibujar estadísticas en el frame: {e}")

            try:
                _, buffer = cv2.imencode('.jpg', frame_procesado)
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(buffer)
            except Exception as e:
                print(f"Error al codificar frame procesado: {e}")

        except Exception as e:
            print(f"Error en el procesamiento del frame: {e}")

    if cap is not None:
        cap.release()
    print("Procesamiento de frames detenido")

def gen():
    while True:
        try:
            buffer = frame_queue.get(timeout=1)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(blank_frame, "Conectando...", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error en generador de video: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    iniciar_hilo_procesamiento()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/estadisticas')
def obtener_estadisticas():
    with lock:
        data = {
            "personas": estadisticas["personas_detectadas"],
            "emocion_predominante": traducir_emocion(estadisticas["emocion_predominante"]),
            "conteo_emociones": {
                traducir_emocion(k): v for k, v in estadisticas["conteo_emociones"].items()
            }
        }
    return jsonify(data)

# Hilo de procesamiento
hilo_procesamiento = None

def iniciar_hilo_procesamiento():
    global hilo_procesamiento
    if hilo_procesamiento is None or not hilo_procesamiento.is_alive():
        hilo_procesamiento = threading.Thread(target=procesar_frames, daemon=True)
        hilo_procesamiento.start()

if __name__ == '__main__':
    app.run(debug=True)
