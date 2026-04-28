import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import random

cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, params)

projeteis = []
explosoes = []

g = 500
vento = random.uniform(-200, 200)

placar_esq = 0
placar_dir = 0

vencedor = None
tempo_vitoria = 0

def criar_projetil(x, y, angulo, dono):
    v = 300
    offset = 25

    x0 = x + offset * math.cos(angulo)
    y0 = y - offset * math.sin(angulo)

    vx = v * math.cos(angulo)
    vy = -v * math.sin(angulo)

    return {
        "x": x0,
        "y": y0,
        "vx": vx,
        "vy": vy,
        "t": time.time(),
        "dono": dono
    }

def criar_explosao(x, y):
    return {
        "x": x,
        "y": y,
        "t": time.time()
    }

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    canhao_esq = None
    canhao_dir = None

    if ids is not None:
        for i in range(len(ids)):
            id = ids[i][0]
            pts = corners[i][0]

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            topo = pts[0]
            dx = topo[0] - cx
            dy = cy - topo[1]
            angulo = math.atan2(dy, dx)

            if id == 0:
                canhao_esq = (cx, cy, angulo)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            elif id == 1:
                canhao_dir = (cx, cy, angulo)
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

    novos_projeteis = []

    for p in projeteis:
        t = time.time() - p["t"]

        if t < 0.2:
            novos_projeteis.append(p)
            continue

        x = int(p["x"] + p["vx"] * t + 0.5 * vento * t * t)
        y = int(p["y"] + p["vy"] * t + 0.5 * g * t * t)

        if 0 < x < w and 0 < y < h:
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

            # colisão
            if canhao_esq and p["dono"] != "esq":
                cx, cy, _ = canhao_esq
                if (x - cx)**2 + (y - cy)**2 < 25**2:
                    placar_dir += 1
                    vencedor = "Direita marcou!"
                    tempo_vitoria = time.time()
                    explosoes.append(criar_explosao(x, y))
                    continue

            if canhao_dir and p["dono"] != "dir":
                cx, cy, _ = canhao_dir
                if (x - cx)**2 + (y - cy)**2 < 25**2:
                    placar_esq += 1
                    vencedor = "Esquerda marcou!"
                    tempo_vitoria = time.time()
                    explosoes.append(criar_explosao(x, y))
                    continue

            novos_projeteis.append(p)

    projeteis = novos_projeteis

    # 💥 ANIMAÇÃO DAS EXPLOSÕES
    novas_explosoes = []
    for e in explosoes:
        t = time.time() - e["t"]

        if t < 0.5:  # duração da explosão
            raio = int(10 + 40 * t)  # cresce com o tempo

            # cor muda (amarelo → vermelho)
            cor = (0, int(255 * (1 - t)), 255)

            cv2.circle(frame, (int(e["x"]), int(e["y"])), raio, cor, 2)
            novas_explosoes.append(e)

    explosoes = novas_explosoes
    frame = cv2.flip(frame, 1)

    # HUD
    cv2.putText(frame, f"{placar_esq} x {placar_dir}", (w//2 - 50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Vento: {int(vento)}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if vencedor:
        cv2.putText(frame, vencedor, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 3)

        if time.time() - tempo_vitoria > 2:
            projeteis = []
            vencedor = None
            vento = random.uniform(-200, 200)

    cv2.imshow("Batalha com Explosoes", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord('a') and canhao_esq:
        x, y, ang = canhao_esq
        projeteis.append(criar_projetil(x, y, ang, "esq"))
    elif key == ord('l') and canhao_dir:
        x, y, ang = canhao_dir
        projeteis.append(criar_projetil(x, y, ang, "dir"))
    elif key == ord('r'):
        placar_esq = 0
        placar_dir = 0
        projeteis = []
        explosoes = []
        vencedor = None

cap.release()
cv2.destroyAllWindows()
