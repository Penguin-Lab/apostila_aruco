import cv2
import numpy as np
import trimesh
import pyrender
from PIL import Image

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
def carregar_obj(caminho,escala=0.01,rot_x=0,rot_y=0,rot_z=0,altura=0.02,max_textura=4096):
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

                        # material.image = np.array(pil_img)
                        material.image = pil_img
                        nw, nh = pil_img.size
                        print(f"Nova textura: {nw}x{nh}")
                    else:
                        img_np = np.fliplr(img_np)
                        # atualiza textura
                        material.image = img_np
                    
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
    "./sistemasolar/sol/sun.obj",
    escala=0.02,
    rot_x=90,
    altura=10
)

# Segundo ArUco ID 1
meshes[1] = carregar_obj(
    "./sistemasolar/mercurio/mercurio.obj",
    escala=0.02,
    rot_x=90,
    altura=2
)

# Terceiro ArUco ID 2
meshes[2] = carregar_obj(
    "./sistemasolar/venus/venus.obj",
    escala=0.03,
    rot_x=90,
    altura=1
)

# Quarto ArUco ID 3
meshes[3] = carregar_obj(
    "./sistemasolar/terra/earth (3).obj",
    escala=0.03,
    rot_x=90,
    altura=1
)

# Quinto ArUco ID 4
meshes[4] = carregar_obj(
    "./sistemasolar/marte/marte_v1.1.obj",
    escala=0.0004,
    rot_x=90,
    altura=100
)

# Sexto ArUco ID 5
meshes[5] = carregar_obj(
    "./sistemasolar/jupiter/realistic_jupiter.obj",
    escala=0.001,
    rot_x=90,
    altura=100
)

# Setimo ArUco ID 6
meshes[6] = carregar_obj(
    "./sistemasolar/saturno/saturno_saturn.obj",
    escala=0.1,
    rot_x=90,
    altura=1,
    max_textura=4096
)

# Oitavo ArUco ID 7
meshes[7] = carregar_obj(
    "./sistemasolar/urano/uranus.obj",
    escala=0.0003,
    rot_x=0,
    altura=400
)

# Nono ArUco ID 8
meshes[8] = carregar_obj(
    "./sistemasolar/netuno/neptune.obj",
    escala=0.04,
    rot_x=90,
    altura=2
)

# Decimo ArUco ID 9
meshes[9] = carregar_obj(
    "./sistemasolar/plutao/pluto.obj",
    escala=0.0002,
    rot_x=90,
    altura=100
)

# Decimo primeioro ArUco ID 10
meshes[10] = carregar_obj(
    "./sistemasolar/lua/lua.obj",
    escala=0.01,
    rot_x=90,
    altura=4
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
            marker_id = int(marker_id)
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

    cv2.imshow("Sistema solar", cv2.flip(frame,1))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# ==========================================
# FINALIZA
# ==========================================
cap.release()
cv2.destroyAllWindows()
