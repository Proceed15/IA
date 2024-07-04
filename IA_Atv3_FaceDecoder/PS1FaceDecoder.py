import sys
import dlib
from skimage import io

# Pegue a imagem (nome.extensão) da linha de comando --> de preferência o caminho completo.
file_name = sys.argv[1]

# Cria uma detector de faces HOG usando a biblioteca nativa dlib.
face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

# Carregue a imagem dentro de uma lista (um array).
image = io.imread(file_name)

# Rode o detector de imagens HOG na imagem.
# O resultado será as bordas em forma de caixa das faces na imagem.
detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Abra uma janela na área de trabalho mostrando a imagem.
win.set_image(image)

# Passa em loop por todas as faces que encontramos na imagem.
for i, face_rect in enumerate(detected_faces):

	# As faces detectadas são retornadas em um object com suas coordenadas...
	# Do topo, da esquerda, da direita e das margens inferiores.
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Desenha uma caixa envolta de todas as faces encontradas;
	win.add_overlay(face_rect)
	        
# Espera até o usuário apertar a tecla Enter para fechar a janela.	        
dlib.hit_enter_to_continue()
