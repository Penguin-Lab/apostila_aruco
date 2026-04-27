import cv2
import cv2.aruco as aruco
import unicodedata

# =========================
# Lista de chamada (ID : Nome)
# =========================
lista_chamada = {
    17: "Állefe Florindo",
    35: "Rafaela Crise",
    49: "Joab Felippe",
    8: "João Pedro",
    9: "Miguel Grigorio"
}

def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

# Conjunto para armazenar presença
presentes = set()

# =========================
# Configuração do ArUco
# =========================
dicionario = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parametros = aruco.DetectorParameters()

# Inicializa a câmera
cap = cv2.VideoCapture(0)

print("Pressione 'q' para encerrar e salvar a lista de presença.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar ArUcos
    cantos, ids, _ = aruco.detectMarkers(gray, dicionario, parameters=parametros)

    if ids is not None:
        # Desenhar os marcadores detectados
        aruco.drawDetectedMarkers(frame, cantos, ids)

        for id in ids.flatten():
            if id in lista_chamada:
                presentes.add(lista_chamada[id])

    # Mostrar imagem
    cv2.imshow("Chamada com ArUco", frame)

    # Sair ao apertar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# =========================
# Salvar presença em arquivo
# =========================
with open("presenca.txt", "w", encoding="utf-8") as f:
    f.write("Lista de Presença:\n\n")
    for nome in sorted(presentes, key=lambda x: remover_acentos(x).lower()):
        f.write(nome + "\n")

print("Presença salva no arquivo presenca.txt")
