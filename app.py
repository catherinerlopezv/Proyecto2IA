from flask import Flask, render_template, Response, jsonify
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variable global
personas_detectadas = 0

def gen_frames():
    global personas_detectadas
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            personas_detectadas = len(faces)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
def estadisticas():
    # Datos de prueba para emociones (rellénalos con tu modelo más adelante)
    datos = {
        "personas": personas_detectadas,
        "emocion_predominante": "---",  # lo llenas luego
        "conteo_emociones": {
            "Feliz": 0,
            "Neutra": 0,
            "Triste": 0
        }
    }
    return jsonify(datos)

if __name__ == '__main__':
    app.run(debug=True)
