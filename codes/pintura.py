# Modificado de Lab Visio: https://github.com/labvisio/kits-demonstracao-visao/blob/main/aruco_draw.py
import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

# Lista de pontos desenhados: (x, y, cor)
desenho = []

# Dicionário ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
params = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, params)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            id = ids[i][0]
            pts = corners[i][0]

            # Centro do marcador
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            # Escolha da ação
            if id == 0:
                desenho.append((cx, cy, (0, 0, 255)))  # vermelho
            elif id == 1:
                desenho.append((cx, cy, (0, 255, 0)))  # verde
            elif id == 2:
                desenho.append((cx, cy, (255, 0, 0)))  # azul
            elif id == 3:
                # Borracha: remove pontos próximos
                raio = 20
                desenho = [p for p in desenho if (p[0]-cx)**2 + (p[1]-cy)**2 > raio**2]

                # Desenha o círculo da borracha na tela
                cv2.circle(frame, (cx, cy), raio, (0, 255, 255), 2)  # amarelo

    # Desenhar tudo
    for x, y, cor in desenho:
        cv2.circle(frame, (x, y), 5, cor, -1)

    frame = cv2.flip(frame, 1)  # 1 = horizontal (efeito espelho)
    cv2.imshow("Desenho com ArUco", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):#27:  # ESC
        break
    elif key == ord('l'):
        desenho = []

cap.release()
cv2.destroyAllWindows()
