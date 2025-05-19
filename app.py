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

# Cargar modelo
print("Cargando modelo...")
modelo = load_model("modelo_emociones90.h5")
print("Modelo cargado correctamente")

# Clases de emociones
clases_emociones = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Inicializar cámara (con tiempo de espera para que la cámara se inicialice correctamente)
def inicializar_camara():
    camara = cv2.VideoCapture(0)
    # Configurar la cámara para menor resolución para mejor rendimiento
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camara.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return None
    return camara

# Detector de rostros
detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Variables globales para estadísticas
estadisticas = {
    "personas_detectadas": 0,
    "emocion_predominante": "---",
    "conteo_emociones": {}
}
lock = threading.Lock()

# Cola para frames procesados
frame_queue = queue.Queue(maxsize=10)
ultima_actualizacion = time.time()
debe_procesar = True

# Función para procesar frames en un hilo separado
def procesar_frames():
    global estadisticas, debe_procesar, ultima_actualizacion
    
    cap = inicializar_camara()
    if cap is None:
        return
    
    while debe_procesar:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame, reintentando...")
            # Reintentar inicializar la cámara
            cap.release()
            time.sleep(1)
            cap = inicializar_camara()
            if cap is None:
                continue
            continue
        
        # Procesar cada 100ms para no sobrecargar la CPU
        tiempo_actual = time.time()
        if tiempo_actual - ultima_actualizacion < 0.1:
            # Agregar un frame sin procesar a la cola para mantener el video fluido
            try:
                if not frame_queue.full():
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_queue.put(buffer)
            except Exception as e:
                print(f"Error al codificar frame: {e}")
            continue
        
        ultima_actualizacion = tiempo_actual
        
        try:
            # Procesar el frame para detección de emociones
            frame_procesado = frame.copy()
            gray = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2GRAY)
            rostros = detector_rostros.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            emociones_detectadas = []
            
            for (x, y, w, h) in rostros:
                try:
                    # Asegurarse de que el rostro está completamente dentro del frame
                    if y < 0 or y + h > frame_procesado.shape[0] or x < 0 or x + w > frame_procesado.shape[1]:
                        continue
                        
                    rostro = gray[y:y + h, x:x + w]
                    if rostro.size == 0:  # Verificar que el rostro no esté vacío
                        continue
                        
                    # Redimensionar al tamaño que espera el modelo (48x48)
                    rostro_redimensionado = cv2.resize(rostro, (48, 48))
                    
                    # Normalizar a valores entre 0 y 1
                    rostro_array = img_to_array(rostro_redimensionado) / 255.0
                    
                    # Agregar dimensión de lote (batch)
                    rostro_array = np.expand_dims(rostro_array, axis=0)
                    
                    # Asegurarse de que el input tenga la forma correcta antes de predecir
                    if rostro_array.shape != (1, 48, 48, 1):
                        rostro_array = rostro_array.reshape(1, 48, 48, 1)
                    
                    # Predecir emoción
                    pred = modelo.predict(rostro_array, verbose=0)[0]
                    clase_idx = np.argmax(pred)
                    emocion = clases_emociones[clase_idx]
                    emoji = get_emoji(emocion)
                    emociones_detectadas.append(emocion)
                    
                    # Dibujar rectángulo y emoción en el frame
                    cv2.rectangle(frame_procesado, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Texto con fondo para mejor legibilidad
                    etiqueta = f"{emoji} {emocion}"
                    text_size = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame_procesado, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 0, 0), -1)
                    cv2.putText(frame_procesado, etiqueta, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                except Exception as e:
                    print(f"Error al procesar rostro: {e}")
                    continue
            
            # Actualizar estadísticas de manera segura
            conteo_actual = {}
            for emo in emociones_detectadas:
                conteo_actual[emo] = conteo_actual.get(emo, 0) + 1
            
            with lock:
                estadisticas["personas_detectadas"] = len(rostros)
                estadisticas["conteo_emociones"] = conteo_actual
                if emociones_detectadas:
                    estadisticas["emocion_predominante"] = max(conteo_actual, key=conteo_actual.get)
                else:
                    estadisticas["emocion_predominante"] = "---"
            
            # Guardar frame procesado en la cola
            try:
                _, buffer = cv2.imencode('.jpg', frame_procesado)
                # Si la cola está llena, sacar un elemento antes de poner el nuevo
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
            # Esperar un frame con timeout para no bloquear
            buffer = frame_queue.get(timeout=1)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            # Si la cola está vacía, entregar un frame en blanco
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
    # Asegurarnos de que el hilo de procesamiento esté activo
    iniciar_hilo_procesamiento()
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/estadisticas')
def obtener_estadisticas():
    with lock:
        data = {
            "personas": estadisticas["personas_detectadas"],
            "emocion_predominante": traducir_emocion(estadisticas["emocion_predominante"]),
            "conteo_emociones": {
                "Feliz": estadisticas["conteo_emociones"].get("happy", 0),
                "Neutra": estadisticas["conteo_emociones"].get("neutral", 0),
                "Triste": estadisticas["conteo_emociones"].get("sad", 0),
                "Enojado": estadisticas["conteo_emociones"].get("angry", 0),
                "Disgusto": estadisticas["conteo_emociones"].get("disgust", 0),
                "Miedo": estadisticas["conteo_emociones"].get("fear", 0),
                "Sorpresa": estadisticas["conteo_emociones"].get("surprise", 0)
            }
        }
    return jsonify(data)

# Función para traducir nombres de emociones al español
def traducir_emocion(emocion):
    traducciones = {
        'happy': 'Feliz',
        'neutral': 'Neutra',
        'sad': 'Triste',
        'angry': 'Enojado',
        'disgust': 'Disgusto',
        'fear': 'Miedo',
        'surprise': 'Sorpresa',
        '---': '---'
    }
    return traducciones.get(emocion, emocion)

# Iniciar el hilo de procesamiento cuando se inicia la aplicación
# Iniciar hilo de procesamiento al arrancar la aplicación
# Flask 2.0+ ya no tiene before_first_request, usamos with_app_context
from flask import current_app

# Variable global para el hilo de procesamiento
hilo_procesamiento = None

# Iniciar el hilo de procesamiento
def iniciar_hilo_procesamiento():
    global hilo_procesamiento
    if hilo_procesamiento is None or not hilo_procesamiento.is_alive():
        hilo_procesamiento = threading.Thread(target=procesar_frames)
        hilo_procesamiento.daemon = True
        hilo_procesamiento.start()
        print("Hilo de procesamiento iniciado")

# Manejar la limpieza cuando la aplicación se cierra
def limpiar():
    global debe_procesar
    debe_procesar = False
    print("Deteniendo procesamiento...")

# Registrar función de limpieza
import atexit
atexit.register(limpiar)

if __name__ == '__main__':
    try:
        # Iniciar el hilo de procesamiento antes de arrancar la aplicación
        iniciar_hilo_procesamiento()
        
        # Usar threaded=True para manejar múltiples conexiones concurrentes
        app.run(debug=False, threaded=True, host='0.0.0.0')
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
        limpiar()