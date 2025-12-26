from flask import Flask, Response, jsonify, send_file
import cv2, os, time, threading
import numpy as np
import sounddevice as sd
from deepface import DeepFace
from ultralytics import YOLO

app = Flask(__name__)

# --- 1. STORAGE CONFIGURATION ---
EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)
MAX_EVIDENCE = 5 

# --- 2. MODEL LOADING ---
yolo = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- 3. GLOBAL STATE ---
current_frame = None
emotion_log = []
# These counts ensure we don't exceed 5 images per type
malpractice_count = {"PHONE": 0, "MULTIPLE_PERSON": 0, "LOOKING_AWAY": 0, "VOICE_HELP": 0}

def map_emotion(raw):
    m = {"neutral": "Neutral", "happy": "Confident", "fear": "Fear", "sad": "Nervousness", "angry": "Nervousness", "surprise": "Hesitation"}
    return m.get(raw, "Neutral")

# --- 4. THE GATEKEEPER FUNCTION ---
def save_evidence(frame, label):
    """
    STRICT STORAGE RULE: 
    This is ONLY called if a violation is detected.
    It stops exactly at 5 images.
    """
    if malpractice_count[label] < MAX_EVIDENCE:
        malpractice_count[label] += 1
        filename = f"{label}_{malpractice_count[label]}.jpg"
        filepath = os.path.join(EVIDENCE_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"[VIOLATION DETECTED] Saved evidence: {filename}")

# --- 5. THE AI ENGINE (Detection Logic) ---
def ai_worker():
    global current_frame, emotion_log
    last_yolo = last_emotion = last_face_seen = time.time()
    
    while True:
        if current_frame is None: 
            time.sleep(0.1); continue
        
        frame = current_frame.copy()
        now = time.time()

        # A. PROCTORING: LOOKING AWAY DETECTION
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        if len(faces) > 0:
            last_face_seen = now
        else:
            # Only store if face is missing for more than 3 seconds
            if now - last_face_seen > 3:
                save_evidence(frame, "LOOKING_AWAY")
                last_face_seen = now # Reset timer to avoid spamming 5 images at once

        # B. PROCTORING: OBJECT DETECTION (PHONE/PERSON)
        if now - last_yolo > 1.2:
            results = yolo(frame, verbose=False, conf=0.1)
            person_count = 0
            phone_found = False
            
            for r in results:
                for box in r.boxes:
                    label = yolo.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    # Store if phone is detected (Low threshold 0.1)
                    if label in ["cell phone", "laptop", "remote"] and conf >= 0.1:
                        phone_found = True
                    
                    # Count persons (High threshold 0.6 to avoid ghosts)
                    if label == "person" and conf >= 0.6:
                        person_count += 1
            
            if phone_found:
                save_evidence(frame, "PHONE")
            if person_count > 1:
                save_evidence(frame, "MULTIPLE_PERSON")
                
            last_yolo = now

        # C. SENTIMENT ANALYSIS (Purely data-driven, no image storage)
        if now - last_emotion > 4:
            try:
                res = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)
                emotion_log.append(map_emotion(res[0]["dominant_emotion"]))
            except: pass
            last_emotion = now
        
        time.sleep(0.01)

# --- 6. AUDIO MONITORING ---
def monitor_noise():
    def callback(indata, frames, time_info, status):
        # Only stores if noise level exceeds threshold
        if np.linalg.norm(indata) * 10 > 0.4:
            if malpractice_count["VOICE_HELP"] < MAX_EVIDENCE:
                # We use a placeholder or black frame for audio evidence if needed, 
                # or just increment the count.
                malpractice_count["VOICE_HELP"] += 1
    with sd.InputStream(callback=callback):
        while True: time.sleep(1)

# Start background threads
threading.Thread(target=ai_worker, daemon=True).start()
threading.Thread(target=monitor_noise, daemon=True).start()

# --- 7. FLASK ROUTES ---
@app.route("/")
def index(): return send_file("index.html")

@app.route("/hr")
def hr_page(): return send_file("hr.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, frame = cap.read()
            if not ret: break
            global current_frame
            current_frame = frame
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/report")
def report():
    t = len(emotion_log) or 1
    cnts = {e: emotion_log.count(e) for e in set(emotion_log)}
    pct = {k: round(v/t*100, 1) for k,v in cnts.items()}
    score = round(pct.get("Neutral", 0) + pct.get("Confident", 0), 1)
    lvl = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
    return jsonify({"emotions": pct, "confidence": score, "level": lvl, "malpractice": malpractice_count})

@app.route("/evidence/<file>")
def get_ev(file): return send_file(os.path.join(EVIDENCE_DIR, file))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)