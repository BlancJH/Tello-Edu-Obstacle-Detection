import threading
import time
import cv2
from flask import Flask, Response

JPEG_QUALITY = 80
FRAME_SIZE = (640, 480)

latest_jpeg = None
frame_lock = threading.Lock()
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Tello MJPEG Stream</h1><img src="/video_feed">'

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame = latest_jpeg
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.01)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_low_level_stream(udp_url="udp://@0.0.0.0:11111"):
    cap = cv2.VideoCapture(udp_url)
    if not cap.isOpened():
        print("Warning: cannot open video stream", udp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.resize(frame, FRAME_SIZE)
        ret2, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ret2:
            with frame_lock:
                global latest_jpeg
                latest_jpeg = jpeg.tobytes()
        time.sleep(0.01)

def mirror_djitellopy_frames(frame_reader):
    import cv2
    global latest_jpeg
    while True:
        frame = frame_reader.frame
        if frame is None:
            time.sleep(0.01)
            continue
        resized = cv2.resize(frame, FRAME_SIZE)
        ret, jpeg = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ret:
            with frame_lock:
                latest_jpeg = jpeg.tobytes()
        time.sleep(0.03)

def start_flask_server(host='0.0.0.0', port=5000):
    threading.Thread(target=lambda: app.run(host=host, port=port, threaded=True), daemon=True).start()
    print(f"🌐 MJPEG server started at http://{host}:{port}")
