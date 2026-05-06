import cv2
import time
import random
import numpy as np
import threading
import queue
import os
from sklearn.cluster import KMeans

# = Modalidade dos Jogos ==
game = 'Demo'
# game = 'Embaralha'

base_path = "./imagem/"     # Pasta das imagens
imagens = [cv2.imread(os.path.join(base_path, f"mapa_{i}.png")) for i in range(1, 16)]

# =========================
# CONFIG WEBCAM
# =========================
cap = cv2.VideoCapture(0)
frame_queue = queue.Queue(maxsize=2)

# === Thread: pegar fotos da camera ===
def tira_foto():
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put_nowait(frame)
        time.sleep(0.01)

# === Thread: Processa e exibe os frames com overlay ===
def processar_frames(id_map):
    cv2.namedWindow("Quebra-Cabeça", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Quebra-Cabeça", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    # habilita tela cheia
    resize_scale = 0.4

    while True:
        try:
            frame_full = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        frame_small = cv2.resize(frame_full, None, fx=resize_scale, fy=resize_scale)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, _ = detector.detectMarkers(gray)

        if markerIds is not None:
            for corners, marker_id in zip(markerCorners, markerIds.flatten()):
                numero_da_peca = id_map.get(int(marker_id))
                
                if numero_da_peca is not None:
                    overlay = imagens[numero_da_peca - 1]
                    if overlay is None: continue
                    h, w = overlay.shape[:2]
                    dst_pts = (corners[0] / resize_scale).astype(np.float32)
                    src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(overlay, matrix, (frame_full.shape[1], frame_full.shape[0]))
                    mask = np.zeros((frame_full.shape[0], frame_full.shape[1]), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
                    mask_3ch = cv2.merge([mask]*3)
                    frame_full = cv2.bitwise_and(frame_full, cv2.bitwise_not(mask_3ch))
                    frame_full = cv2.add(frame_full, cv2.bitwise_and(warped, mask_3ch))

        frame_resized = cv2.resize(frame_full, None, fx=3, fy=3)
        cv2.imshow("Quebra-Cabeça", cv2.flip(frame_resized,1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# === THREAD: VERIFICA GRADE DE ARUCOS ========
def verificar_grade():
    last_match_state = None

    while True:
        time.sleep(0.5)
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) < 15:
            continue

        # lógica de homografia e grade
        side_len = 150.0 
        output_size = (int(side_len * 10), int(side_len * 10))
        pts_src = all_corners[0][0]
        offset = np.array([output_size[0] / 2 - side_len / 2, output_size[1] / 2 - side_len / 2], dtype="float32")
        pts_dst = np.array([
            [offset[0], offset[1]], [offset[0] + side_len - 1, offset[1]],
            [offset[0] + side_len - 1, offset[1] + side_len - 1], [offset[0], offset[1] + side_len - 1]
        ], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_image = cv2.warpPerspective(frame, matrix, output_size)
        gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        warped_corners, warped_ids, _ = detector.detectMarkers(gray_warped)

        if warped_ids is None or len(warped_ids) < 15:
            continue

        grid_detectada = [[None for _ in range(4)] for _ in range(4)]
        warped_centers = np.array([c[0].mean(axis=0) for c in warped_corners])
        marker_ids_list = warped_ids.flatten()
        kmeans_rows = KMeans(n_clusters=4, random_state=0, n_init=10).fit(warped_centers[:, 1].reshape(-1, 1))
        row_labels = kmeans_rows.labels_
        row_centers_sorted = np.argsort(kmeans_rows.cluster_centers_[:, 0])
        row_map = {label: i for i, label in enumerate(row_centers_sorted)}
        kmeans_cols = KMeans(n_clusters=4, random_state=0, n_init=10).fit(warped_centers[:, 0].reshape(-1, 1))
        col_labels = kmeans_cols.labels_
        col_centers_sorted = np.argsort(kmeans_cols.cluster_centers_[:, 0])
        col_map = {label: i for i, label in enumerate(col_centers_sorted)}

        for i, marker_id in enumerate(marker_ids_list):
            row = row_map[row_labels[i]]
            col = col_map[col_labels[i]]
            if 0 <= row < 4 and 0 <= col < 4:
                grid_detectada[row][col] = marker_id

        match = (expected == grid_detectada)

        if match:
            print("Sequência correta!")
        else:
            print("Sequência incorreta!")

if game == 'Demo':
    expected = [[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12], [13,14,15,None]]
else:
    numeros = list(range(1, 16))
    random.shuffle(numeros)
    numeros.append(None)
    expected = [numeros[i*4:(i+1)*4] for i in range(4)]
    print("Grade esperada:")
    for row in expected:
        print(row)

id_para_peca_map = {}
peca_numero = 1
for r in range(4):
    for c in range(4):
        if peca_numero > 15: break
        id_aruco = expected[r][c]
        if id_aruco is not None:
            id_para_peca_map[id_aruco] = peca_numero
        peca_numero += 1

parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

frame_queue = queue.Queue(maxsize=2)

t1 = threading.Thread(target=tira_foto, daemon=True)
t2 = threading.Thread(target=processar_frames, args=(id_para_peca_map,))
t3 = threading.Thread(target=verificar_grade, daemon=True)

for t in [t1, t2, t3]: t.start()
t2.join()
