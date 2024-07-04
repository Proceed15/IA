import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from twilio.rest import Client


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


# Twilio, as informações da conta utilizada
twilio_account_sid = 'SEU_SID_TWILIO'
twilio_auth_token = 'SEU_TOKEN_DE_AUTENTICACAO_TWILIO'
twilio_phone_number = 'SEU_TELEFONE_PRINCIPAL_TWILIO'
destination_phone_number = 'TELEFONE_PARA_NOTIFICACAO_SMS'
client = Client(twilio_account_sid, twilio_auth_token)


# Diretório raiz para o projeto, escolha o seu
ROOT_DIR_PATH = Path(".")

# Diretório para salvar os registros e o modelo treinado
MODEL_DIR = os.path.join(ROOT_DIR_PATH, "logs")

# Endereço para o diretório local dos arquivos de pesos de treinamento
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Faz Download dos pesos de treinamento COCO de Releases se necessário
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Diretório das imagens para se rodar as detecções
IMAGE_DIR = os.path.join(ROOT_DIR_PATH, "imagens")

# Endereço até o arquivo de vídeo ou camera para processamento
# Se pretende usar sua Webcam em tempo real, deixe o valor como 0
VIDEO_SOURCE = "suaPasta/seuArquivo.mp4"

# Crie uma (máscara) Mask-RCNN como modelo em modo de inferência
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Carregue um modelo pré-treinado
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Locais das vagas para se estacionar
parked_car_boxes = None

# Carrega o arquivo de vídeo em que se pode rodar a detecção
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Para Contar quantos frames (trechos) de vídeos foram vistos em sequencia
# contendo uma vaga livre
free_space_frames = 0

# Para Verificar se o SMS já foi enviado
sms_sent = False

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

    if parked_car_boxes is None:
        # Esse é o primeiro frame do vídeo, assume que todos os carros detectados
        # estão estacionados nas vagas
        # Salva a localização de cada carro como um cubo com suas dimensões respectivas,
        # em resumo uma caixa. Vai ao próximo frame de vídeo
        parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
    else:
        # Como já se sabe onde estão essa vagas...
        # Checa se há atualmente (nesse frame) alguma vaga desocupada

        # Seleciona as bordas onde estão carros atualmente
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])

        # Vê o quanto os carros estão ocupando em outras vagas próximas conhecidas
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

        # Assume que não há vagas livres até encontrar alguma
        free_space = False

        # Passa em loop por cada caixa conhecida de uma vaga
        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

            # Para esta vaga, encontre o máximo que foi coberta por qualquer
            # carro que foi detectado na imagem (não importa qual carro especificamente)
            max_IoU_overlap = np.max(overlap_areas)

            # Pega as coordenadas da esquerda, direita, acima e debaixo da vaga
            y1, x1, y2, x2 = parking_area

            # Checa se o espaço das vagas está ocupado a partir da checagem
            # se o espaço da vaga está ocupado a mais que o.15 usando o IoU
            if max_IoU_overlap < 0.15:
                # Espaço das vagas desocupado, desenha uma caixa verde envolta
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Mostra que viu pelo menos uma vaga livre
                free_space = True
            else:
                # Espaço das vagas ainda ocupado, desenha uma caixa vermelha envolta
                Parking space is still occupied - draw a red box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Escreve a metragem IoU na caixa
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

        # Se um espaço(vaga) está livre, começa a contar frames
        # Assim não alertamos baseado em um frame de um espaço aberto...
        # Auxilia o algoritmo ser bem sucedido retornando vagas disponíveis reais
        if free_space:
            free_space_frames += 1
        else:
            # Se nenhuma vaga está livre, reseta a contagem
            free_space_frames = 0

        # Se o espaço está livre por vários frames (alguns segundos),
        # Temos algo próximo de certeza que a vaga está livre
        if free_space_frames > 10:
            # Escreve VAGA ENCONTRADA! no topo da tela
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"VAGA ENCONTRADA!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

            # Se um SMS ainda não foi enviado, envia
            if not sms_sent:
                print("ENVIANDO SMS!!!")
                message = client.messages.create(
                    body="Vaga Diponível - vai vai vai!",
                    from_=twilio_phone_number,
                    to=destination_phone_number
                )
                sms_sent = True

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
