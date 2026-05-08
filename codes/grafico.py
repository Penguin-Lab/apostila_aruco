import cv2
import cv2.aruco as aruco
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================
REFERENCE_ID = 1

REFERENCE_MARKER_SIZE = 0.21   # 21 cm
OBJECT_MARKER_SIZE = 0.071     # 7.1 cm

show_grid = True
xy_plane_only = True

# ============================================================
# CÂMERA
# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Erro ao abrir câmera.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Erro ao capturar frame.")
    exit()

h, w = frame.shape[:2]

# ============================================================
# MATRIZ INTRÍNSECA APROXIMADA
# ============================================================
focal_length = w
camera_matrix = np.array([
    [focal_length, 0, w / 2],
    [0, focal_length, h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# ============================================================
# ARUCO
# ============================================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = (aruco.CORNER_REFINE_SUBPIX)
detector = aruco.ArucoDetector(aruco_dict,parameters)

# ============================================================
# MODELOS 3D
# ============================================================
ref_half = REFERENCE_MARKER_SIZE / 2
reference_obj_points = np.array([
    [-ref_half,  ref_half, 0],
    [ ref_half,  ref_half, 0],
    [ ref_half, -ref_half, 0],
    [-ref_half, -ref_half, 0]
], dtype=np.float32)

obj_half = OBJECT_MARKER_SIZE / 2
object_obj_points = np.array([
    [-obj_half,  obj_half, 0],
    [ obj_half,  obj_half, 0],
    [ obj_half, -obj_half, 0],
    [-obj_half, -obj_half, 0]
], dtype=np.float32)

# ============================================================
# MATRIZ HOMOGÊNEA
# ============================================================
def transformation_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

# ============================================================
# GRID 3D
# ============================================================
def draw_3d_grid(frame,rvec,tvec,grid_size=0.5,step=0.1,xy_only=False):
    R, _ = cv2.Rodrigues(rvec)
    values = np.arange(-grid_size,grid_size + step,step)
    lines = []

    # ========================================================
    # PLANO XY
    # ========================================================
    for y in values:
        lines.append((np.array([-grid_size, y, 0]),np.array([ grid_size, y, 0])))

    for x in values:
        lines.append((np.array([x, -grid_size, 0]),np.array([x,  grid_size, 0])))

    # ========================================================
    # GRID 3D COMPLETO
    # ========================================================
    if not xy_only:
        # XZ
        for z in values:
            lines.append((np.array([-grid_size, 0, z]),np.array([ grid_size, 0, z])))
        for x in values:
            lines.append((np.array([x, 0, -grid_size]),np.array([x, 0,  grid_size])))

        # YZ
        for z in values:
            lines.append((np.array([0, -grid_size, z]),np.array([0,  grid_size, z])))
        for y in values:
            lines.append((np.array([0, y, -grid_size]),np.array([0, y,  grid_size])))

    # ========================================================
    # DESENHO
    # ========================================================
    for p1, p2 in lines:
        pts_3d = np.float32([p1, p2])
        cam_pts = ((R @ pts_3d.T).T + tvec.reshape(1, 3))

        z1 = cam_pts[0][2]
        z2 = cam_pts[1][2]

        if z1 <= 0.01 or z2 <= 0.01:
            continue

        pts_2d, _ = cv2.projectPoints(pts_3d,rvec,tvec,camera_matrix,dist_coeffs)
        pts_2d = pts_2d.reshape(-1, 2)

        pt1 = tuple(np.int32(pts_2d[0]))
        pt2 = tuple(np.int32(pts_2d[1]))

        if (abs(pt1[0]) > 5000 or abs(pt1[1]) > 5000 or abs(pt2[0]) > 5000 or abs(pt2[1]) > 5000):
            continue

        cv2.line(frame,pt1,pt2,(90, 90, 90),1,cv2.LINE_AA)

# ============================================================
# EIXOS
# ============================================================
def draw_axes(frame,rvec,tvec,size=0.10):
    axis_points_3D = np.float32([
        [0, 0, 0],
        [size, 0, 0],
        [0, -size, 0],
        [0, 0, size]
    ])

    imgpts, _ = cv2.projectPoints(axis_points_3D,rvec,tvec,camera_matrix,dist_coeffs)
    imgpts = imgpts.astype(int).reshape(-1, 2)
    origin = tuple(imgpts[0])

    # X vermelho
    cv2.line(frame,origin,tuple(imgpts[1] * 8 - 7 * imgpts[0]),(0,0,155),3)
    # Y verde
    cv2.line(frame,origin,tuple(imgpts[2] * 8 - 7 * imgpts[0]),(0,155,0),3)
    # Z azul
    cv2.line(frame,origin,tuple(imgpts[3] * 8 - 7 * imgpts[0]),(155,0,0),3)

# ============================================================
# FILTRO
# ============================================================
alpha = 0.7
filtered_positions = {}

# ============================================================
# LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    reference_pose = None
    marker_poses = {}

    # ========================================================
    # DETECÇÃO
    # ========================================================
    if ids is not None:
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            img_points = corners[i][0].astype(np.float32)
            if marker_id == REFERENCE_ID:
                current_obj_points = reference_obj_points
            else:
                current_obj_points = object_obj_points
            success, rvec, tvec = cv2.solvePnP(
                current_obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            rvec = rvec.reshape(3, 1).astype(np.float32)
            tvec = tvec.reshape(3, 1).astype(np.float32)

            marker_poses[marker_id] = (rvec,tvec,img_points)

            # =================================================
            # REFERENCIAL
            # =================================================
            if marker_id == REFERENCE_ID:
                reference_pose = (rvec,tvec)

                if show_grid:
                    draw_3d_grid(frame,rvec,tvec,grid_size=0.5,step=0.1,xy_only=xy_plane_only)

                draw_axes(frame,rvec,tvec,size=0.10)

    # ========================================================
    # FRAME ESPELHADO
    # ========================================================
    display_frame = cv2.flip(frame.copy(),1)

    # ========================================================
    # DESENHO DOS OBJETOS
    # ========================================================
    for marker_id, pose_data in marker_poses.items():
        rvec, tvec, img_points = pose_data
        center = np.mean(img_points,axis=0).astype(int)
        mirrored_center = (w - center[0],center[1])

        # ====================================================
        # REFERENCIAL
        # ====================================================
        if marker_id == REFERENCE_ID:
            cv2.circle(display_frame,mirrored_center,8,(0,155,0),-1)
            cv2.putText(display_frame,"REFERENCIAL",
                (mirrored_center[0] - 70,mirrored_center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,155,0),2)

        # ====================================================
        # OBJETOS
        # ====================================================
        else:
            if reference_pose is not None:
                ref_rvec, ref_tvec = reference_pose
                T_ref = transformation_matrix(ref_rvec,ref_tvec)
                T_obj = transformation_matrix(rvec,tvec)
                T_relative = (np.linalg.inv(T_ref) @ T_obj)

                position = T_relative[:3, 3]
                if marker_id not in filtered_positions:
                    filtered_positions[marker_id] = position
                else:
                    filtered_positions[marker_id] = (alpha*filtered_positions[marker_id] + (1 - alpha)*position)

                x = filtered_positions[marker_id][0]
                y = -filtered_positions[marker_id][1]
                z = filtered_positions[marker_id][2]

                cv2.circle(display_frame,mirrored_center,8,(0,0,155),-1)
                text1 = f"ID {marker_id}"
                text2 = (f"({x:.2f},"f"{y:.2f},"f"{z:.2f})")
                cv2.putText(display_frame,text1,
                    (mirrored_center[0] - 40,mirrored_center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(155,0,0),2)
                cv2.putText(display_frame,text2,
                    (mirrored_center[0] - 100,mirrored_center[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,155,0),2)

    # ========================================================
    # EXIBE
    # ========================================================
    cv2.imshow("Coordenadas com ArUcos",display_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # ========================================================
    # GRID ON/OFF
    # ========================================================
    elif key == ord('g'):
        show_grid = not show_grid
        print(f"Grid: {'ON' if show_grid else 'OFF'}")

    # ========================================================
    # XY ONLY
    # ========================================================
    elif key == ord('c'):
        xy_plane_only = not xy_plane_only
        print("Modo Grid:","XY_ONLY"if xy_plane_only else "3D")

# ============================================================
# FINALIZAÇÃO
# ============================================================
cap.release()
cv2.destroyAllWindows()
