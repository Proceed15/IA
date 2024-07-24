import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3

# Esse Programa usa um modelo da Keras do Google para identificar
# um objeto e mostra com quanta fidelidade pôde chegar a essa conclusão

# Carrega o modelo: Google’s Inception v3 deep neural network da Keras
# esse modelo pode identificar 1000 tipos de objetos pré-treinadas
model = inception_v3.InceptionV3()

# Carrega o arquivo da imagem e o converte em um array para o numpy utilizar
img = image.load_img("cat.png", target_size=(299, 299))
input_image = image.img_to_array(img)

# Redimensiona a imagem para que a intensidade de todos os pixels
# varie entre [-1, 1] como o modelo espera receber o objeto
input_image /= 255.
input_image -= 0.5
input_image *= 2.

# Adiciona uma quarta dimensão para o tamanho desse lote,
# como Keras espera receber a imagem assim, é enviado aqui
input_image = np.expand_dims(input_image, axis=0)

# Atravessa a imagem pela rede neural
predictions = model.predict(input_image)

# Converte as predições em texto e as printa na tela
predicted_classes = inception_v3.decode_predictions(predictions, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
#Mensagem com o objeto identificado e a porcentagem de fidelidade
print("Isso é um(a) {} com {:.4}% de fidelidade!".format(name, confidence * 100))

