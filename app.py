from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Cargar modelo entrenado
modelo_emociones = load_model('modelo_emociones90.h5')
emociones = ['Enojo', 'Disgusto', 'Miedo', 'Feliz', 'Neutra', 'Triste', 'Sorpresa']  # Usa las emociones reales según tu entrenamiento

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

estadisticas = {
    "personas": 0,
    "emocion_predominante": "---",
    "conteo_emociones": {emo: 0 for emo in emociones}
}

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            emocion_contador = {emo: 0 for emo in emociones}
            estadisticas["personas"] = len(faces)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = gray[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (48, 48))
                rostro = rostro.astype("float") / 255.0
                rostro = img_to_array(rostro)
                rostro = np.expand_dims(rostro, axis=0)

                pred = modelo_emociones.predict(rostro, verbose=0)[0]
                emocion_idx = np.argmax(pred)
                emocion = emociones[emocion_idx]

                emocion_contador[emocion] += 1
                cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            estadisticas["conteo_emociones"] = emocion_contador
            if sum(emocion_contador.values()) > 0:
                estadisticas["emocion_predominante"] = max(emocion_contador, key=emocion_contador.get)
            else:
                estadisticas["emocion_predominante"] = "---"

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/estadisticas')
def estadisticas_endpoint():
    return jsonify(estadisticas)

if __name__ == '__main__':
    app.run(debug=True)
