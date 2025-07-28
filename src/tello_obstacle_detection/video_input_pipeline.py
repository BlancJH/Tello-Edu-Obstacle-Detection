import cv2
import socket
import threading
from flask import Flask, Response
import time

# 1. SETUP: Activate Tello video stream (command + streamon) This coded in low level to improve performance.

tello_address = ('192.168.10.1', 8889)
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock.bind(('', 9000))  # for receiving responses

def send_cmd(msg):
    command_sock.sendto(msg.encode(), tello_address)

send_cmd('command')
time.sleep(1)
send_cmd('streamon')
time.sleep(1)

# 2. CAPTURE: OpenCV reads from Tello's raw H.264 UDP stream

cap = cv2.VideoCapture("udp://@0.0.0.0:11111")

# Shared variable for latest frame
latest_jpeg = None
lock = threading.Lock()

# 3. BACKGROUND THREAD: Read + encode frames continuously
def capture_frames():
    global latest_jpeg
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Resize if needed
        frame = cv2.resize(frame, (640, 480))

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with lock:
                latest_jpeg = jpeg.tobytes()

# Start capture thread
threading.Thread(target=capture_frames, daemon=True).start()

# 4. FLASK SERVER: Serve MJPEG stream at http://localhost:5000
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Tello MJPEG Stream</h1><img src="/video_feed">'

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                frame = latest_jpeg
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.01)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 5. RUN SERVER
if __name__ == '__main__':
    print("🌐 Visit http://localhost:5000 to view Tello MJPEG stream")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        cap.release()
        send_cmd('streamoff')
        command_sock.close()
