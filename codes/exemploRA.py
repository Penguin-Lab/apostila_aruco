import cv2
import numpy as np
import trimesh
import pyrender

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width = frame.shape[:2]

# ==========================================
# ARUCO
# ==========================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ==========================================
# MATRIZ DA CAMERA
# ==========================================
fx = width
fy = width

cx = width / 2
cy = height / 2

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))

# ==========================================
# TAMANHO REAL DO ARUCO
# ==========================================
marker_size = 0.05

obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# ==========================================
# CENA
# ==========================================
scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

# ==========================================
# CAMERA 3D
# ==========================================
camera = pyrender.IntrinsicsCamera(fx=fx,fy=fy,cx=cx,cy=cy)
cam_node = scene.add(camera)

# ==========================================
# LUZ
# ==========================================
light = pyrender.DirectionalLight(color=np.ones(3),intensity=3.0)
scene.add(light)

# ==========================================
# RENDERER
# ==========================================
renderer = pyrender.OffscreenRenderer(width,height)

# ==========================================
# FUNCAO PARA PREPARAR OBJETOS
# ==========================================
def carregar_obj(caminho,escala=0.01,rot_x=0,rot_y=0,rot_z=0,altura=0.02):
    mesh = trimesh.load(caminho)

    # centraliza
    mesh.apply_translation(-mesh.centroid)

    # ==========================
    # ROTACOES
    # ==========================
    rx = trimesh.transformations.rotation_matrix(np.radians(rot_x),[1, 0, 0])
    ry = trimesh.transformations.rotation_matrix(np.radians(rot_y),[0, 1, 0])
    rz = trimesh.transformations.rotation_matrix(np.radians(rot_z),[0, 0, 1])

    mesh.apply_transform(rx)
    mesh.apply_transform(ry)
    mesh.apply_transform(rz)
    
    # ==========================
    # ESCALA
    # ==========================
    mesh.apply_scale(escala)

    # ==========================
    # MOVE PRA CIMA
    # ==========================
    mesh.apply_translation([0, 0, altura])

    return pyrender.Mesh.from_trimesh(mesh)

# ==========================================
# CARREGA OBJETOS
# ==========================================
meshes = {}

# Primeiro ArUco ID 0
meshes[0] = carregar_obj(
    "./cenario/fox/low-poly-fox-by-pixelmannen.obj",
    escala=0.0005,
    rot_x=90
)

# Segundo ArUco ID 20
meshes[20] = carregar_obj(
    "./cenario/fox/low-poly-fox-by-pixelmannen.obj",
    escala=0.001,
    rot_x=90
)

# Terceiro ArUco ID 3
meshes[3] = carregar_obj(
    "./cenario/fox/low-poly-fox-by-pixelmannen.obj",
    escala=0.002,
    rot_x=90
)

# ==========================================
# CRIA NODES FIXOS
# ==========================================
nodes = {}
for marker_id, mesh in meshes.items():
    node = scene.add(mesh,pose=np.eye(4))
    nodes[marker_id] = node

# ==========================================
# LOOP
# ==========================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    # ======================================
    # ESCONDE OBJETOS
    # ======================================
    for node in nodes.values():
        scene.set_pose(node,
            np.array([
                [1,0,0,1000],
                [0,1,0,1000],
                [0,0,1,1000],
                [0,0,0,1]
            ])
        )

    # ======================================
    # DETECCAO
    # ======================================
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame,corners,ids)
        for i, marker_id in enumerate(ids.flatten()):
            img_points = corners[i][0]
            success, rvec, tvec = cv2.solvePnP(obj_points,img_points,camera_matrix,dist_coeffs)
            if success:
                rot_mat, _ = cv2.Rodrigues(rvec)
                pose = np.eye(4)
                pose[:3, :3] = rot_mat
                pose[:3, 3] = tvec.flatten()

                # ==================================
                # CORRIGE EIXOS OPENGL
                # ==================================
                pose[1, :] *= -1
                pose[2, :] *= -1

                # ==================================
                # ATUALIZA POSE DO NODE
                # ==================================
                if marker_id in nodes:
                    scene.set_pose(nodes[marker_id],pose)

    # ======================================
    # RENDERIZA UMA VEZ
    # ======================================
    color, depth = renderer.render(scene)

    # RGB -> BGR
    color = color[:, :, ::-1]

    # mascara
    mask = depth > 0

    # mistura
    frame[mask] = color[:, :, :3][mask]

    cv2.imshow("Cenario RA", cv2.flip(frame,1))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# ==========================================
# FINALIZA
# ==========================================
cap.release()
cv2.destroyAllWindows()
