import sys
import dlib
from skimage import io

# Você pode baixar e instaar o modelo de detecção de faces (do dlib), já pré-treinado e necessário a essa atividade, aqui:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Irá automaticamente baixar o arquivo.

predictor_model = "shape_predictor_68_face_landmarks.dat"

# Pega o nome do arquivo de imagem da linha de comando (CMD prompt).
file_name = sys.argv[1]

# Cria o detector de faces HOG usando a já implatada classe do dlib (por isso e mais é importante instalá-la em seu computador antes).
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

# Pega o nome do arquivo de imagem da linha de comando (CMD prompt).
file_name = sys.argv[1]

# Carrega a imagem.
image = io.imread(file_name)

# Usa o detector de faces HOG nos dados da imagem.
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# Mostra a janela na área de trabalho com a imagem.
win.set_image(image)

# Faz uma verificação (loop) perante cada face que encontrou na imagem (até encontrar alguma).
for i, face_rect in enumerate(detected_faces):

	# As faces identificadas são retornadas como um objeto com... 
	# coordenadas da parte de cima, esquerda, direita e partes inferiores.
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Desenha uma caixa envolta de cada face identificada.
	win.add_overlay(face_rect)

	# Pega a posição em que a face está (de lado, centralizada).
	pose_landmarks = face_pose_predictor(image, face_rect)

	# Encontra os marcos do rosto na tela (nesse caso 68, podem ser até 93).
	win.add_overlay(pose_landmarks)
	        
dlib.hit_enter_to_continue()
