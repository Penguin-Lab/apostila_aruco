import cv2, numpy as np, pygame

# =========================
# ARQUIVOS
# =========================
SND_OPEN  = './cofre/abriu.mp3'
SND_CLICK = './cofre/click.mp3'
SND_RIGHT = './cofre/right_angle.mp3'

IMG_LOCK  = './cofre/combinacao.png'
IMG_FINAL = './cofre/seven.png'

# =========================
# FUNÇÕES
# =========================
def angulo(p1, p2):
    ang = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
    return ang + 360 if ang < 0 else ang

def subangulo(angvelho,angnovo):
    distancia = (angnovo - angvelho + 180) % 360 - 180
    return distancia

def direcao(delta,dir,limiar):
    if abs(delta) > 1:
        if (np.sign(delta) != dir and abs(delta) > limiar) or (np.sign(delta) == dir): 
            return  np.sign(delta)
    return 0

# =========================
# MAIN
# =========================
pygame.init(); pygame.mixer.init()
pygame.mixer.set_num_channels(8)

ch_click = pygame.mixer.Channel(0)
ch_right = pygame.mixer.Channel(1)
ch_open  = pygame.mixer.Channel(2)

click  = pygame.mixer.Sound(SND_CLICK)
open_s = pygame.mixer.Sound(SND_OPEN)
right  = pygame.mixer.Sound(SND_RIGHT)

lock  = cv2.imread(IMG_LOCK)
final = cv2.resize(cv2.imread(IMG_FINAL), (900,900))

h, w = lock.shape[:2]
src = np.float32([[0,0],[w,0],[w,h],[0,h]])

cap = cv2.VideoCapture(0)

aruco = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters()
)

seq_dir = [1,-1,1,-1]
seq_ang = [90,180,270,0]
LIM = 8

state = 0
last_ang = 0
mostrar_final = False

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (900,900))
    out = frame.copy()

    corners, ids, _ = aruco.detectMarkers(frame)

    if ids is not None:
        for pts, i in zip(corners, ids):
            if i == 8:
                pts = pts[0]

                ang = angulo(pts[0], pts[1])
                delta = subangulo(last_ang,ang)

                d = direcao(delta,seq_dir[state],LIM)

                # verifica direção
                if d != 0 and mostrar_final != True:
                    last_ang = ang
                    click.play()
                    if d != seq_dir[state]:
                        state = 0
                        print("Errou!")
                    else:
                        # verifica ângulo
                        if abs(ang - seq_ang[state]) < 6:
                            if state == 3:
                                ch_open.play(open_s)
                                mostrar_final = True
                                print("Abriu!")
                            else:
                                ch_right.play(right)
                                print("Proximo!")
                                state += 1

                # overlay imagem no marcador
                H, _ = cv2.findHomography(src, pts)
                warp = cv2.warpPerspective(lock, H, (900,900))

                mask = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                bg = cv2.bitwise_and(out,out,mask=mask_inv)
                fg = cv2.bitwise_and(warp,warp,mask=mask)

                out = cv2.add(bg, fg)

    # tela final
    if mostrar_final:
        cv2.imshow("frame", final)
    else:    
        cv2.imshow("frame", cv2.flip(out,1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
