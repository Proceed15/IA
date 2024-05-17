import sys
import dlib
import cv2
import openface

# Você pode baixar e instaar o modelo de detecção de faces (do dlib), já pré-treinado e necessário a essa atividade, aqui:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Irá automaticamente baixar o arquivo, deixei o zip caso prefira baixar desse repositório nesta pasta.
predictor_model = "shape_predictor_68_face_landmarks.dat"

# Pega o nome do arquivo de imagem da linha de comando (CMD prompt).
file_name = sys.argv[1]

# Cria o detector de faces HOG usando a já implatada classe do dlib (por isso e mais é importante instalá-la em seu computador antes).
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

# Pega o nome do arquivo de imagem da linha de comando (CMD prompt).
file_name = sys.argv[1]

# Carrega a imagem.
image = cv2.imread(file_name)

# Usa o detector de faces HOG nos dados da imagem.
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# Mostra a janela na área de trabalho com a imagem.
for i, face_rect in enumerate(detected_faces):

	# As faces identificadas são retornadas como um objeto com... 
	# coordenadas da parte de cima, esquerda, direita e partes inferiores.
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Pega a posição em que a face está (de lado, centralizada).
	pose_landmarks = face_pose_predictor(image, face_rect)
	
	# Nesta etapa precisamos ter utilizado o algoritmo do FaceFinder para encontrar os marcos do rosto na imagem.
	# Usa openface para calcular os marcos do rosto e fazer um alinhamento centralizado desta face.
	alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

	# Salva a imagem alinhada em um arquivo .jpg.
	cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)
	        
