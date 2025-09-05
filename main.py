import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading
import queue

# TTS
def tts_worker(q: queue.Queue, stop_event: threading.Event):
    engine = pyttsx3.init()
    # inicia o loop do pyttsx3 sem bloquear a thread
    engine.startLoop(False)

    try:
        while not stop_event.is_set():
            # pega novo texto sem bloquear muito (permite iterar o motor)
            try:
                new_text = q.get(timeout=0.05)
                if new_text == "__QUIT__":
                    break
                # queremos falar só o último pedido (sem fila): interrompe o que houver
                engine.stop()            # limpa fala/queue anterior
                engine.say(new_text)     # agenda a fala mais recente
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


model = YOLO('yolov8n.pt')  # modelo YOLO
cap = cv2.VideoCapture(1)   # câmera

last_spoken = {}            # controla cooldown por classe
cooldown = 5                # segundos
near_threshold = 0.10       # fração da tela para ser "perto"

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

        # Seleciona somente o objeto "mais próximo" (maior fração de área)
        closest_obj = None
        closest_fraction = 0.0
        region = None

        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = box.int().tolist()
            area = max(1, (x2 - x1)) * max(1, (y2 - y1))
            fraction = area / frame_area
            cls_name = results.names[int(cls_id)]

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

            # guarda o maior
            if fraction > closest_fraction:
                closest_fraction = fraction
                closest_obj = cls_name
                region = obj_region

        # desenha boxes (opcional: todos ou só o mais próximo)
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = box.int().tolist()
            cls_name = results.names[int(cls_id)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # fala apenas se o mais próximo realmente está "perto"
        now = time.time()
        if closest_obj and closest_fraction > near_threshold:
            # respeita cooldown por classe
            if closest_obj not in last_spoken or now - last_spoken[closest_obj] > cooldown:
                text = f'{closest_obj} {region}'
                tts_queue.put(text)
                last_spoken[closest_obj] = now

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