import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

# Configuração: ID -> (caminho PNG, escala)
imagens = {
    0: ("mona.png", 2.0),
    1: ("grito.png", 2.0),
    2: ("noite.png", 2.0)
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

def overlay_png(frame, png, corners, escala=1.0):

    h_img, w_img = png.shape[:2]
    h_img2, w_img2 = h_img/2, w_img/2

    # ==========================
    # CENTRO DO PNG COM QUADRADO
    # ==========================
    centro_src = np.array([w_img2, h_img2], dtype=np.float32)
    d_png = min(w_img2, h_img2)/escala
    pts_src = np.array([
        [centro_src[0]+d_png,centro_src[1]-d_png],
        [centro_src[0]-d_png,centro_src[1]-d_png],
        [centro_src[0]-d_png,centro_src[1]+d_png],
        [centro_src[0]+d_png,centro_src[1]+d_png]
    ], dtype=np.float32)

    # ==========================
    # ARUCO QUADRADO
    # ==========================
    pts_dst = np.array(corners, dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # ==========================
    # IMAGEM
    # ==========================
    if png.shape[2] == 4:
        rgb = png[:, :, :3]
        alpha = png[:, :, 3] / 255.0
    else:
        rgb = png
        alpha = np.ones((h_img, w_img))

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
