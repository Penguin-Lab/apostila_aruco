import os
from tkinter import Tk, filedialog
from PIL import Image

# Oculta a janela principal do tkinter
root = Tk()
root.withdraw()

# Abre seletor de arquivo
file_path = filedialog.askopenfilename(
    title="Selecione uma imagem quadrada",
    filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")]
)

if not file_path:
    print("Nenhum arquivo selecionado.")
    exit()

# Abre imagem
img = Image.open(file_path)
width, height = img.size

# Verifica se é quadrada
if width != height:
    print("A imagem não é quadrada!")
    exit()

# Define tamanho de cada corte (4x4)
tile_size = width // 4

# Pasta onde será salvo
output_dir = os.path.dirname(file_path)

# Nome base do arquivo
base_name = os.path.splitext(os.path.basename(file_path))[0]

count = 0

# Loop para cortar 4x4
for i in range(4):
    for j in range(4):
        left = j * tile_size
        upper = i * tile_size
        right = left + tile_size
        lower = upper + tile_size

        tile = img.crop((left, upper, right, lower))

        output_path = os.path.join(output_dir, f"{base_name}_{count+1}.png")
        tile.save(output_path)

        count += 1

print("Imagem cortada em 16 partes com sucesso!")
