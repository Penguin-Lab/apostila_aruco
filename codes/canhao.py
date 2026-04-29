import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import random

# =========================
# CONFIGURAÇÕES
# =========================
ID_TANQUE_verm = 88
ID_TANQUE_azul = 973

GRAVIDADE = 500
COOLDOWN = 0.5
DANO = 25
HP_MAX = 100

cap = cv2.VideoCapture(0)

detector = aruco.ArucoDetector(
    aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
    aruco.DetectorParameters()
)

# =========================
# ESTADO DO JOGO
# =========================
tanques = {"verm": None, "azul": None}
projeteis = []
explosoes = []
placar = {"verm": 0, "azul": 0}

vento = random.uniform(-200, 200)
vencedor = None
tempo_vitoria = 2


# =========================
# CRIAÇÃO DE OBJETOS
# =========================
def criar_tanque(x, y, angulo):
    return {
        "x": x,
        "y": y,
        "ang": angulo,
        "hp": HP_MAX,
        "ultimo_tiro": 0
    }


def criar_projetil(tanque, dono):
    ang = tanque["ang"]
    v = 300

    return {
        "x": tanque["x"] + 35 * math.cos(ang),
        "y": tanque["y"] - 35 * math.sin(ang),
        "vx": v * math.cos(ang),
        "vy": -v * math.sin(ang),
        "tempo": time.time(),
        "dono": dono
    }


def criar_explosao(x, y):
    return {"x": x, "y": y, "tempo": time.time()}


# =========================
# DESENHO
# =========================
def desenhar_tanque(frame, t, cor):
    x, y, ang = int(t["x"]), int(t["y"]), t["ang"]

    # corpo
    corpo = np.array([[-25, -15], [25, -15], [25, 15], [-25, 15]])
    R = np.array([[math.cos(ang), -math.sin(ang)],
                  [math.sin(ang),  math.cos(ang)]])

    pts = np.dot(corpo, R.T) + [x, y]
    pts = pts.astype(int)

    cv2.fillPoly(frame, [pts], cor)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

    # canhão
    ponta = (int(x + 35 * math.cos(ang)),
             int(y - 35 * math.sin(ang)))
    cv2.line(frame, (x, y), ponta, (40, 40, 40), 6)

    # barra de vida
    hp = int(50 * max(t["hp"], 0) / HP_MAX)
    cv2.rectangle(frame, (x-25, y-38), (x+25, y-30), (50,50,50), -1)
    cv2.rectangle(frame, (x-25, y-38), (x-25+hp, y-30), (0,255,0), -1)


# =========================
# DETECÇÃO DOS ARUCOS
# =========================
def atualizar_tanques(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return

    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]

        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()

        dx = pts[0][0] - cx
        dy = cy - pts[0][1]
        ang = math.atan2(dy, dx)

        if marker_id == ID_TANQUE_verm:
            lado = "verm"
        elif marker_id == ID_TANQUE_azul:
            lado = "azul"
        else:
            continue

        if tanques[lado] is None:
            tanques[lado] = criar_tanque(cx, cy, ang)
        else:
            tanques[lado].update(x=cx, y=cy, ang=ang)


# =========================
# FÍSICA DOS PROJÉTEIS
# =========================
def atualizar_projeteis(frame, largura, altura):
    global vencedor, tempo_vitoria

    novos = []

    for p in projeteis:
        t = time.time() - p["tempo"]

        if t < 0.2:
            novos.append(p)
            continue

        x = int(p["x"] + p["vx"]*t + 0.5*vento*t*t)
        y = int(p["y"] + p["vy"]*t + 0.5*GRAVIDADE*t*t)

        if 0 < x < largura and 0 < y < altura:
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

            # colisão com tanques
            for lado in ["verm", "azul"]:
                tnk = tanques[lado]

                if not tnk or tnk["hp"] <= 0 or p["dono"] == lado:
                    continue

                if (x - tnk["x"])**2 + (y - tnk["y"])**2 < 25**2:
                    tnk["hp"] -= DANO
                    explosoes.append(criar_explosao(x, y))

                    if tnk["hp"] <= 0:
                        outro = "azul" if lado == "verm" else "verm"
                        placar[outro] += 1
                        vencedor = outro
                        tempo_vitoria = time.time()

                    break
            else:
                novos.append(p)

    return novos


# =========================
# EXPLOSÕES
# =========================
def atualizar_explosoes(frame):
    novas = []

    for e in explosoes:
        t = time.time() - e["tempo"]

        if t < 0.5:
            raio = int(10 + 40*t)
            cor = (0, int(255*(1-t)), 255)
            cv2.circle(frame, (int(e["x"]), int(e["y"])), raio, cor, 2)
            novas.append(e)

    return novas


# =========================
# LOOP PRINCIPAL
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    atualizar_tanques(frame)

    # desenhar tanques
    if tanques["verm"] and tanques["verm"]["hp"] > 0:
        desenhar_tanque(frame, tanques["verm"], (0, 0, 255))

    if tanques["azul"] and tanques["azul"]["hp"] > 0:
        desenhar_tanque(frame, tanques["azul"], (255, 0, 0))

    # atualizar jogo
    projeteis = atualizar_projeteis(frame, w, h)
    explosoes = atualizar_explosoes(frame)

    frame = cv2.flip(frame, 1)

    # HUD
    cv2.putText(frame, f"{placar['verm']} x {placar['azul']}",
                (w//2 - 50, 40), 0, 1, (255,255,255), 2)

    cv2.putText(frame, f"Vento: {int(vento)}",
                (20, h-20), 0, 0.7, (200,200,200), 2)

    # reset após vitória
    if vencedor:
        if vencedor == "verm":
            cv2.putText(frame, "Vermelho ganhou!", (w//2 - 140, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Azul ganhou!", (w//2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if time.time() - tempo_vitoria > 2:
            projeteis.clear()
            vento = random.uniform(-200, 200)
            vencedor = None

            for t in tanques.values():
                if t:
                    t["hp"] = HP_MAX

    cv2.imshow("Batalha de Tanques", frame)

    # =========================
    # CONTROLES
    # =========================
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('a') and tanques["verm"]:
        t = tanques["verm"]
        if time.time() - t["ultimo_tiro"] > COOLDOWN:
            projeteis.append(criar_projetil(t, "verm"))
            t["ultimo_tiro"] = time.time()

    if key == ord('l') and tanques["azul"]:
        t = tanques["azul"]
        if time.time() - t["ultimo_tiro"] > COOLDOWN:
            projeteis.append(criar_projetil(t, "azul"))
            t["ultimo_tiro"] = time.time()

    if key == ord('r'):
        placar = {"verm": 0, "azul": 0}
        projeteis.clear()
        explosoes.clear()
        vento = random.uniform(-200, 200)

        for t in tanques.values():
            if t:
                t["hp"] = HP_MAX

cap.release()
cv2.destroyAllWindows()
