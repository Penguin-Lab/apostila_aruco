import cv2
import cv2.aruco as aruco
import numpy as np
import time

# =========================
# CONFIGURAÇÃO
# =========================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

pares = {
    973: "mona.png", 62: "mona.png",
    118: "noite.png", 88: "noite.png",
    49: "grito.png", 17: "grito.png"
}

imagens = {k: cv2.imread(v) for k, v in pares.items()}

fixos = set()

# controle de tentativas (vários pares ao mesmo tempo)
tentativas = []  
# cada tentativa: {"ids": [id1, id2], "tempo": t}

# =========================
# LOOP PRINCIPAL
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    detectados = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in pares and marker_id not in fixos:
                detectados.append(marker_id)

    # =========================
    # CRIAR NOVAS TENTATIVAS
    # =========================
    usados = set()
    for i in range(len(detectados)):
        for j in range(i+1, len(detectados)):
            id1, id2 = detectados[i], detectados[j]

            if id1 in usados or id2 in usados:
                continue

            # evita duplicar tentativa
            ja_existe = any(set(t["ids"]) == {id1, id2} for t in tentativas)

            if not ja_existe:
                tentativas.append({
                    "ids": [id1, id2],
                    "tempo": time.time()
                })
                usados.update([id1, id2])

    # =========================
    # PROCESSAR TENTATIVAS
    # =========================
    novas_tentativas = []
    revelados = set()

    for t in tentativas:
        id1, id2 = t["ids"]

        if pares[id1] == pares[id2]:
            fixos.update([id1, id2])
        elif time.time() - t["tempo"] < 2:
            novas_tentativas.append(t)
            revelados.update([id1, id2])

    tentativas = novas_tentativas

    # =========================
    # DESENHO
    # =========================
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0].astype(int)

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            if marker_id in fixos or marker_id in revelados:
                marker_id = int(marker_id)
                img = imagens[marker_id]
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    x1, y1 = cx - 50, cy - 50
                    x2, y2 = cx + 50, cy + 50

                    if 0 < x1 and 0 < y1 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                        frame[y1:y2, x1:x2] = img
            else:
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

    frame = cv2.flip(frame, 1)
    # =========================
    # VITÓRIA
    # =========================
    if len(fixos) == len(pares):
        cv2.putText(frame, "Voce venceu!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.namedWindow("Jogo da Memoria com ArUco", cv2.WINDOW_NORMAL)
    cv2.imshow("Jogo da Memoria com ArUco", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
