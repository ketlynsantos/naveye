import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading
import queue
from translations import class_translation

# Configurações de prioridade
priority_classes = ['person', 'car', 'bicycle', 'dog', 'motorcycle', 'bus', 'stop sign', 'cat', 'train', 'boat', 'traffic light', 'chair', 'couch', 'bed', 'bench', 'table', 'backpack', 'suitcase']
min_confidence = 0.5
max_objects_to_speak = 3
base_cooldown = 5 # segundos
near_threshold = 0.10 # fração da tela para ser "perto"

# TTS
def tts_worker(q: queue.Queue, stop_event: threading.Event):
    engine = pyttsx3.init()
    # inicia o loop do pyttsx3 sem bloquear a thread
    engine.startLoop(False)

    current_text = None

    try:
        while not stop_event.is_set():
            # pega novo texto sem bloquear muito (permite iterar o motor)
            try:
                new_text = q.get(timeout=0.05)
                if new_text == "__QUIT__":
                    break
                    
                if new_text != current_text: 
                    # queremos falar só o último pedido (sem fila): interrompe o que houver
                    engine.stop() # limpa fala/queue anterior
                    engine.say(new_text) # agenda a fala mais recente
                    current_text = new_text
            except queue.Empty:
                pass

            # bombeia o loop interno do engine
            engine.iterate()

        # antes de sair, esvazia e encerra
        engine.stop()
    finally:
        try:
            engine.endLoop()
        except Exception:
            pass


model = YOLO('yolov8n.pt') # modelo YOLO
cap = cv2.VideoCapture(1) # câmera

last_spoken = {} # controla cooldown por classe

# fila e thread do TTS
tts_queue = queue.Queue()
stop_event = threading.Event()
tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, stop_event), daemon=True)
tts_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        frame_area = height * width

        results = model(frame)[0]

        detected_objs = []

        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if conf < min_confidence:
                continue

            cls_name = results.names[int(cls_id)]
            if cls_name not in priority_classes:
                continue

            x1, y1, x2, y2 = box.int().tolist()
            area = max(1, (x2 - x1)) * max(1, (y2 - y1))
            fraction = area / frame_area

            # posição central para decidir região
            cx = (x1 + x2) / 2
            if cx < (width / 5):
                obj_region = 'à esquerda'
            elif cx < (width / 2.5):
                obj_region = 'frente-esquerda'
            elif cx < (3 * width / 5):
                obj_region = 'à frente'
            elif cx < (4 * width / 5):
                obj_region = 'frente-direita'
            else:
                obj_region = 'à direita'

            detected_objs.append({
                'cls_name': cls_name,
                'fraction': fraction,
                'region': obj_region
            })

        # desenha boxes (opcional: todos ou só o mais próximo)
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = box.int().tolist()
            cls_name = results.names[int(cls_id)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Ordena por proximidade e fala até N objetos
        detected_objs.sort(key=lambda d: d['fraction'], reverse=True)
        for obj in detected_objs[:max_objects_to_speak]:
            now = time.time()
            # cooldown variável: mais próximo, fala mais rápido
            cooldown = base_cooldown * (1 - min(obj['fraction']/0.5,0.9))
            key = (obj['cls_name'], obj['region'])

            if key not in last_spoken or now - last_spoken[key] > cooldown:
                translate_cls_name = class_translation.get(obj['cls_name'])
                text = f'{translate_cls_name} {obj['region']}'
                tts_queue.put(text)
                last_spoken[key] = now

        cv2.imshow('YOLO + Voz', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # encerra TTS com segurança
    stop_event.set()
    tts_queue.put("__QUIT__")
    tts_thread.join()