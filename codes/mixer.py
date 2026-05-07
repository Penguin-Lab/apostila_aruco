import cv2
import pygame
import numpy as np

# ==========================================
# INICIA AUDIO
# ==========================================
pygame.mixer.init()
pygame.mixer.set_num_channels(16)
canais = {}

for i in range(16):
    canais[i] = pygame.mixer.Channel(i)

# ==========================================
# CARREGA SONS
# ==========================================
sons = {}
# 0 - fox
# 1 - floresta
# 2 - praia
# 3 - passaros
# 4 - chuva
# 5 - fogueira
# 6 - vila
# 7 - barco

sons[0] = pygame.mixer.Sound("./sons/fox.wav")
sons[1] = pygame.mixer.Sound("./sons/floresta.wav")
sons[2] = pygame.mixer.Sound("./sons/praia.wav")
sons[3] = pygame.mixer.Sound("./sons/passaros.wav")
sons[4] = pygame.mixer.Sound("./sons/chuva.wav")
sons[5] = pygame.mixer.Sound("./sons/fogueira.wav")
sons[6] = pygame.mixer.Sound("./sons/vila_medieval.wav")
sons[7] = pygame.mixer.Sound("./sons/barco.wav")

# volumes
sons[0].set_volume(0.6)
sons[1].set_volume(0.6)
sons[2].set_volume(0.5)
sons[3].set_volume(0.5)
sons[4].set_volume(0.5)
sons[5].set_volume(0.5)
sons[6].set_volume(0.5)
sons[7].set_volume(0.5)

# ==========================================
# SONS TOCANDO
# ==========================================
tocando = {}

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

# ==========================================
# ARUCO
# ==========================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ==========================================
# LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    corners, ids, _ = detector.detectMarkers(frame)

    # ======================================
    # ARUCOS VISIVEIS
    # ======================================
    ids_ativos = set()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame,corners,ids)
        ids_ativos = set(ids.flatten())

        # ==================================
        # TOCA NOVOS SONS
        # ==================================
        for marker_id in ids_ativos:
            if marker_id in sons:
                if marker_id not in tocando:
                    canal = canais[marker_id]

                    canal.play(sons[marker_id],loops=-1)
                    tocando[marker_id] = canal

                    print(f"Tocando som do ArUco {marker_id}")

    # ======================================
    # PARA SONS REMOVIDOS
    # ======================================
    for marker_id in list(tocando.keys()):
        if marker_id not in ids_ativos:
            tocando[marker_id].stop()
            del tocando[marker_id]
            print(f"Parando som do ArUco {marker_id}")

    # ======================================
    # MOSTRA CAMERA
    # ======================================
    cv2.imshow("Audio ArUco",cv2.flip(frame,1))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# ==========================================
# FINALIZA
# ==========================================
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
