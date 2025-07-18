# app.py
import cv2
import numpy as np
import pyttsx3
import threading
import time
from datetime import datetime
import os
from ultralytics import YOLO
import streamlit as st
from collections import defaultdict, deque
import queue as qu

# ----------------------------------------------------------
# 1.  Stable detection helper
# ----------------------------------------------------------
class StableDetectionTracker:
    def __init__(self, stability_frames=5, confidence_threshold=0.7):
        self.stability_frames = stability_frames
        self.confidence_threshold = confidence_threshold
        self.detection_history = deque(maxlen=stability_frames)
        self.object_lifetimes = defaultdict(int)

    def update(self, detections):
        self.detection_history.append(detections)
        object_counts = defaultdict(list)
        for frame_detections in self.detection_history:
            seen = set()
            for det in frame_detections:
                key = det['name']
                object_counts[key].append(det)
                seen.add(key)
            for obj in seen:
                self.object_lifetimes[obj] += 1

        stable = []
        for name, dets in object_counts.items():
            if len(dets) >= self.stability_frames * self.confidence_threshold:
                best = max(dets, key=lambda x: x['confidence'])
                best['lifetime'] = self.object_lifetimes[name]
                stable.append(best)
        return stable

# ----------------------------------------------------------
# 2.  Assistant class
# ----------------------------------------------------------
class SmartDetectionAssistant:
    def __init__(self):
        self.init_model()
        self.init_tts()
        self.init_settings()
        self.start_tts_worker()

    # --- model ---
    def init_model(self):
        if 'model' not in st.session_state:
            with st.spinner("Loading YOLOv8 model..."):
                st.session_state.model = YOLO('yolov8n.pt')
        self.model = st.session_state.model

    # --- TTS engine ---
    def init_tts(self):
        if 'tts_engine' not in st.session_state:
            st.session_state.tts_engine = pyttsx3.init()
            st.session_state.tts_engine.setProperty('rate', 160)
            st.session_state.tts_engine.setProperty('volume', 0.9)
            voices = st.session_state.tts_engine.getProperty('voices')
            if voices:
                st.session_state.tts_engine.setProperty('voice', voices[0].id)
        self.tts_engine = st.session_state.tts_engine
        self.tts_queue = qu.Queue()
        self.tts_busy = False

    # --- settings ---
    def init_settings(self):
        self.confidence_threshold = 0.35
        self.tracker = StableDetectionTracker(stability_frames=3, confidence_threshold=0.6)
        self.last_speech_time = 0
        self.speech_interval = 3.0

    # --- TTS worker thread (once per session) ---
    def start_tts_worker(self):
        if 'tts_thread_started' not in st.session_state:
            st.session_state.tts_thread_started = False
        if not st.session_state.tts_thread_started:
            def worker():
                while True:
                    try:
                        text = self.tts_queue.get(timeout=1)
                        if text is None:
                            break
                        self.tts_busy = True
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                        self.tts_busy = False
                        self.tts_queue.task_done()
                    except qu.Empty:
                        continue
                    except Exception as e:
                        st.error(f"TTS Error: {e}")
                        self.tts_busy = False
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            st.session_state.tts_thread_started = True

    # --- speak ---
    def speak_async(self, text):
        if not self.tts_busy:
            try:
                self.tts_queue.put(text, block=False)
                return True
            except qu.Full:
                return False
        return False

    # --- detection ---
    def detect_objects(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, conf=self.confidence_threshold, iou=0.5)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    area = (x2 - x1) * (y2 - y1)
                    if area < 1000:
                        continue
                    name = self.model.names[cls]
                    detections.append({
                        'name': name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': area
                    })
        return detections

    # --- draw ---
    def draw_stable_detections(self, frame, stable):
        out = frame.copy()
        for det in stable:
            bbox = det['bbox']
            name = det['name']
            conf = det['confidence']
            life = det.get('lifetime', 0)

            color = (0, 255, 0) if life > 10 else ((0, 255, 255) if life > 5 else (0, 165, 255))
            thick = 3 if life > 5 else 2
            cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)

            label = f"{name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (bbox[0], bbox[1]-h-15), (bbox[0]+w+10, bbox[1]), color, -1)
            cv2.putText(out, label, (bbox[0]+5, bbox[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    # --- speech text ---
    def create_speech_text(self, stable):
        if not stable:
            return None
        counts = {}
        for d in stable:
            counts[d['name']] = counts.get(d['name'], 0) + 1
        parts = [f"{cnt} {n}s" if cnt > 1 else f"a {n}" for n, cnt in counts.items()]
        if len(parts) == 1:
            return f"I can see {parts[0]}."
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}."
        else:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}."

# ----------------------------------------------------------
# 3.  Camera background thread
# ----------------------------------------------------------
def camera_worker(stop_event, frame_queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while not stop_event.is_set():
        ok, img = cap.read()
        if ok:
            try:
                frame_queue.put_nowait(img)
            except qu.Full:
                pass  # drop oldest
        else:
            time.sleep(0.01)
    cap.release()

# ----------------------------------------------------------
# 4.  Streamlit UI
# ----------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Smart Object Detection Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    .main {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);}
    .stApp {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);}
    .metric-card{background:rgba(255,255,255,.15);backdrop-filter:blur(10px);border-radius:10px;padding:15px;text-align:center;margin:5px;}
    .status-indicator{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:8px;}
    .status-active{background:#4CAF50}.status-inactive{background:#f44336}.status-processing{background:#ff9800}
    </style>
    """, unsafe_allow_html=True)

    # Session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = SmartDetectionAssistant()
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    assistant = st.session_state.assistant

    st.title("🤖 Smart Object Detection Assistant")
    st.markdown("**Real-time AI-powered object detection with voice feedback**")

    col1, col2 = st.columns([2, 1])

    # ------------- SIDEBAR -------------
    with st.sidebar:
        st.header("🎛️ Controls")
        st.subheader("Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        assistant.confidence_threshold = confidence
        stability_frames = st.slider("Stability Frames", 1, 10, 3)
        assistant.tracker.stability_frames = stability_frames

        st.subheader("Voice Settings")
        enable_voice = st.checkbox("Enable Voice Feedback", value=True)
        if enable_voice:
            speech_interval = st.slider("Speech Interval (s)", 1.0, 10.0, 3.0, 0.5)
            assistant.speech_interval = speech_interval
            if st.button("🔊 Test Voice"):
                assistant.speak_async("Voice system is working correctly!")

        st.subheader("Manual Controls")
        if st.button("🔄 Reset Detection History"):
            assistant.tracker = StableDetectionTracker(stability_frames, 0.6)
            st.session_state.detection_history = []
            st.success("Detection history reset!")
        if st.button("📊 Export Detection Log"):
            if st.session_state.detection_history:
                lines = ["timestamp,objects,count"]
                lines += [f"{e['timestamp']},{', '.join(e['objects'])},{e['count']}"
                          for e in st.session_state.detection_history]
                st.download_button(
                    label="📥 Download CSV",
                    data="\n".join(lines),
                    file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # ------------- MAIN COLUMN -------------
    with col1:
        st.subheader("📹 Live Detection Feed")
        placeholder = st.empty()

        if st.button("🎥 Start Camera"):
            st.session_state["stop_cam"] = threading.Event()
            st.session_state["frame_q"] = qu.Queue(maxsize=5)
            th = threading.Thread(
                target=camera_worker,
                args=(st.session_state["stop_cam"], st.session_state["frame_q"]),
                daemon=True
            )
            th.start()
            st.session_state["cam_live"] = True

        if st.button("⏹️ Stop Camera"):
            if st.session_state.get("cam_live"):
                st.session_state["stop_cam"].set()
                st.session_state["cam_live"] = False
                placeholder.empty()

        if st.session_state.get("cam_live", False):
            while True:
                try:
                    frame = st.session_state["frame_q"].get_nowait()
                except qu.Empty:
                    continue

                detections = assistant.detect_objects(frame)
                stable = assistant.tracker.update(detections)
                annotated = assistant.draw_stable_detections(frame, stable)

                placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

                # voice
                if enable_voice and stable:
                    now = time.time()
                    if now - assistant.last_speech_time > assistant.speech_interval:
                        txt = assistant.create_speech_text(stable)
                        if txt and assistant.speak_async(txt):
                            assistant.last_speech_time = now

                # history
                if stable:
                    st.session_state.detection_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "objects": [d["name"] for d in stable],
                        "count": len(stable)
                    })
                    st.session_state.detection_history = st.session_state.detection_history[-50:]
                    st.session_state.total_detections = len(stable)

    # ------------- RIGHT COLUMN -------------
    with col2:
        st.subheader("📊 Detection Analytics")
        status_html = f"""
        <div class="metric-card">
            <div class="status-indicator {'status-active' if st.session_state.get('cam_live', False) else 'status-inactive'}"></div>
            Camera: {'Active' if st.session_state.get('cam_live', False) else 'Inactive'}
        </div>
        <div class="metric-card">
            <div class="status-indicator {'status-active' if enable_voice else 'status-inactive'}"></div>
            Voice: {'Enabled' if enable_voice else 'Disabled'}
        </div>
        <div class="metric-card">
            <div class="status-indicator {'status-processing' if assistant.tts_busy else 'status-active'}"></div>
            TTS: {'Speaking' if assistant.tts_busy else 'Ready'}
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)

        st.markdown("### Current Detections")
        st.metric("Objects Detected", st.session_state.total_detections)

        st.markdown("### Recent Activity")
        if st.session_state.detection_history:
            for entry in st.session_state.detection_history[-10:]:
                with st.expander(f"🕐 {entry['timestamp']} – {entry['count']} objects"):
                    for obj in entry["objects"]:
                        st.write(f"• {obj}")
        else:
            st.info("No detection history yet")

        if st.session_state.detection_history:
            st.markdown("### Statistics")
            from collections import Counter
            counts = Counter(o for e in st.session_state.detection_history for o in e["objects"])
            for obj, cnt in counts.most_common(5):
                st.write(f"• {obj}: {cnt} times")

# ----------------------------------------------------------
if __name__ == "__main__":
    main()