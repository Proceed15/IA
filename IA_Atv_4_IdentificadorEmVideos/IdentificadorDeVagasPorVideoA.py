# Algoritmo para identificação de vagas livres em uma região a partir de uma filmagem da mesma
# VideoParkingIdentificatorA
# É retornado uma imagem quando há uma vaga sem um carro estacionado
# Usa Machine Learning para identificação em tempo real nas filmagens
import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


# Configuração que será usada pelo recurso (nesse caso uma library) Mask-RCNN
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # O dataset COCO tem 80 classes e 1 classe à background
    # COCO é uma sigla para : Common Objects In Context. Um dataset para detecção, segmentação e captação de objetos em imagens e vídeos
    DETECTION_MIN_CONFIDENCE = 0.6


# Filtra uma lista de máscaras (masks) R-CNN de resultados para conseguir somente os carros ou caminhões que forem detectados na gravação
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # Se o objeto detectado não for um carro ou caminhão, pula para o próximo
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Diretório raiz para o projeto, escolha o seu
ROOT_DIR = Path(".")

# Diretório para salvar os registros e o modelo treinado
MODEL_DIR = os.path.join(ROOT_DIR, "registros")

# Endereço para o diretório local dos arquivos de pesos de treinamento
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Faz Download dos pesos de treinamento COCO de Releases se necessário
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Diretório das imagens para se rodar as detecções
IMAGE_DIR = os.path.join(ROOT_DIR, "imagens")

# Endereço até o arquivo de vídeo ou camera para processamento
# Se pretende usar sua Webcam em tempo real, deixe o valor como 0
VIDEO_SOURCE = "suaPasta/seuArquivo.mp4"

# Cria uma (máscara) Mask-RCNN como modelo em modo de inferência
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Carregue um modelo pré-treinado
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Locais das vagas para se estacionar
parked_car_boxes = None

# Carrega o arquivo de vídeo em que se pode rodar a detecção
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Passa em loop por cada frame do vídeo
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Converte a imagem de cores BGR (que o OpenCV usa) para cores RGB 
    rgb_image = frame[:, :, ::-1]

    # Roda a imagem pela Máscara do modelo R-CNN para obter os resultados
    results = model.detect([rgb_image], verbose=0)

    # Como a máscara R-CNN assume que estamos rodando a detecção em várias imagens
    # Só foi passada uma imagem para detecção, então com o código abaixo só pegará
    # O primeiro resltado no índice zero.
    r = results[0]

    # A variável r agora terá os resultado para detecção com:
    # + r['rois'] são as bordas de cada caixa contornando cada objeto detectado
    # + r['class_ids'] é a classe id (tipo) de cada objeto detectado
    # + r['scores'] são as chances do objeto detectado ser o que é descrito, quanto mais altas melhores
    # + r['masks'] são as máscaras de objeto para (for) cada objeto detectado, o que possibilitam os contornos

    # Filtra os resultados para só selecionar as bordas do carro / caminhão
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    print("Carros encontrados em frame do vídeo:")

    # Desenha cada caixa para os frames correspondentes
    for box in car_boxes:
        print("Car: ", box)

        y1, x1, y2, x2 = box

        # Desenha a caixa
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Revela o frame do vídeo na tela a sendo exibida
    cv2.imshow('Vídeo', frame)

    # Aperte 'q' para sair
    # Esse if contém o código ASCII da tecla q
    # Substitua juntamente ao q em questão se desejar usar outra tecla
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpa tudo quando encerrado
video_capture.release()
cv2.destroyAllWindows()
