import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

# Configuração: ID -> (caminho PNG, escala)
imagens = {
    0: ("mona.png", 10.0),
    1: ("grito.png", 5),
    2: ("noite.png", 6)
}

# carregar imagens (com alpha)
imagens_carregadas = {}
for k, (path, escala) in imagens.items():
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # mantém transparência
    imagens_carregadas[k] = (img, escala)

# ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, params)

def overlay_png(frame, png, corners, escala):
    h_img, w_img = png.shape[:2]

    # dimensões do ArUco (aproximadas)
    largura_aruco = np.linalg.norm(corners[0] - corners[1])
    altura_aruco = np.linalg.norm(corners[1] - corners[2])

    # aplicar escala no alvo
    largura_aruco *= escala
    altura_aruco *= escala

    # manter proporção (fit dentro do ArUco)
    fator = min(largura_aruco / w_img, altura_aruco / h_img)

    new_w = int(w_img * fator)
    new_h = int(h_img * fator)

    png = cv2.resize(png, (new_w, new_h))

    # centralizar dentro do ArUco
    cx = np.mean(corners[:, 0])
    cy = np.mean(corners[:, 1])

    pts_dst = np.array([
        [cx - new_w/2, cy - new_h/2],
        [cx + new_w/2, cy - new_h/2],
        [cx + new_w/2, cy + new_h/2],
        [cx - new_w/2, cy + new_h/2]
    ], dtype=np.float32)

    pts_src = np.array([
        [0, 0],
        [new_w, 0],
        [new_w, new_h],
        [0, new_h]
    ], dtype=np.float32)

    # homografia
    M, _ = cv2.findHomography(pts_src, pts_dst)

    # alpha
    if png.shape[2] == 4:
        rgb = png[:, :, :3]
        alpha = png[:, :, 3] / 255.0
    else:
        rgb = png
        alpha = np.ones((new_h, new_w))

    warped = cv2.warpPerspective(rgb, M, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpPerspective(alpha, M, (frame.shape[1], frame.shape[0]))

    for c in range(3):
        frame[:, :, c] = frame[:, :, c] * (1 - mask) + warped[:, :, c] * mask

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            id = ids[i][0]

            if id in imagens_carregadas:
                png, escala = imagens_carregadas[id]
                frame = overlay_png(frame, png, corners[i][0], escala)

    frame = cv2.flip(frame, 1)
    cv2.imshow("Substituicao ArUco", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
