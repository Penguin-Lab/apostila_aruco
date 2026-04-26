import cv2

# Carrega o dicionario de ArUco (voce pode mudar se quiser outro tipo)
dicionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Parametros do detector
parametros = cv2.aruco.DetectorParameters()

# Cria o detector
detector = cv2.aruco.ArucoDetector(dicionario, parametros)

# Abre a camera (0 = camera padrao do notebook)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem")
        break

    # Detecta os ArUcos
    corners, ids, _ = detector.detectMarkers(frame)

    # Se encontrou algum marcador
    if ids is not None:
        # Desenha os contornos e IDs
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Mostra a imagem
    cv2.imshow("Deteccao de ArUco", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a camera e fecha tudo
cap.release()
cv2.destroyAllWindows()
