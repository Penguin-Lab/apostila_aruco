import cv2
import pygame
import numpy as np
import trimesh
import pyrender
from PIL import Image

# ==========================================
# INICIA AUDIO
# ==========================================
pygame.mixer.init()
pygame.mixer.set_num_channels(16)
canais = {}

for i in range(16):
    canais[i] = pygame.mixer.Channel(i)

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
light = pyrender.DirectionalLight(color=np.ones(3),intensity=20.0)
scene.add(light)

# ==========================================
# RENDERER
# ==========================================
renderer = pyrender.OffscreenRenderer(width,height)

# ==========================================
# FUNCAO PARA PREPARAR OBJETOS
# ==========================================
def carregar_obj(caminho,escala=0.01,rot_x=0,rot_y=0,rot_z=0,altura=0.02,max_textura=2048):
    mesh = trimesh.load(caminho)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )

    try:
        if hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            if hasattr(material, 'image'):
                img = material.image
                if img is not None:
                    img_np = np.array(img)
                    h, w = img_np.shape[:2]
                    print(f"Textura original: {w}x{h}")
                    # se textura muito grande
                    if w > max_textura or h > max_textura:
                        print("Redimensionando textura...")
                        pil_img = Image.fromarray(img_np)
                        pil_img.thumbnail((max_textura, max_textura))

                        material.image = np.array(pil_img)
                        nh, nw = material.image.shape[:2]
                        print(f"Nova textura: {nw}x{nh}")
    except Exception as e:
        print("Erro ao processar textura:")
        print(e)

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
    # MOVE PRA CIMA
    # ==========================
    mesh.apply_translation([0, 0, altura])

    # ==========================
    # ESCALA
    # ==========================
    mesh.apply_scale(escala)

    return pyrender.Mesh.from_trimesh(mesh)

# ==========================================
# CARREGA OBJETOS
# ==========================================
meshes = {}

# Primeiro ArUco ID 0
meshes[0] = carregar_obj(
    "./cenario/fox/low-poly-fox-by-pixelmannen.obj",
    escala=0.0005,
    rot_x=90,
    altura=40
)

# Segundo ArUco ID 1
meshes[1] = carregar_obj(
    "./cenario/rio/low_poly_river.obj",
    escala=0.02,
    rot_x=90,
    altura=-3
)

# Terceiro ArUco ID 2
meshes[2] = carregar_obj(
    "./cenario/praia/the_beach.obj",
    escala=0.02,
    rot_x=90,
    altura=0
)

# Quarto ArUco ID 3
meshes[3] = carregar_obj(
    "./cenario/birds/birds.obj",
    escala=0.02,
    rot_x=180,
    altura=5
)

# Quinto ArUco ID 4
meshes[4] = carregar_obj(
    "./cenario/chuva/rain_1.obj",
    escala=0.0002,
    rot_x=90,
    altura=1000
)

# Sexto ArUco ID 5
meshes[5] = carregar_obj(
    "./cenario/fogueira/printable_fire_pit.obj",
    escala=0.01,
    rot_x=90,
    altura=0
)

# Setimo ArUco ID 6
meshes[6] = carregar_obj(
    "./cenario/vila/village_low_poly.obj",
    escala=0.1,
    rot_x=90,
    altura=0,
    max_textura=4096
)

# Oitavo ArUco ID 7
meshes[7] = carregar_obj(
    "./cenario/barco/rowing_boat.obj",
    escala=0.00002,
    rot_x=90,
    altura=1000
)

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
    ids_ativos = set()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame,corners,ids)
        ids_ativos = set(ids.flatten())
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
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
