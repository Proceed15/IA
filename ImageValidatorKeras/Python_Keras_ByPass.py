import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image

# Esse Algoritmo usa um modelo da Keras do Google para identificar um objeto e
# mostra com quanta fidelidade pôde chegar a essa conclusão, com esse algoritmo foi
# possível passar a imagem de um gato ao modelo como se fosse de uma torradeira,
# Usando sistema de Classes, Alteração de Gradientes e Pixels na imagem original

# Carrega o modelo: Google’s Inception v3 deep neural network da Keras
# esse modelo pode identificar 1000 tipos de objetos pré-treinadas
model = inception_v3.InceptionV3()

# Pega uma referência da primeira e última camada (layer) da rede neural
model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

# Escolha um objeto das classes ImmageNet para simular
# Uma lista das classes está disponível nesse link em .json:
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
# Usa-se a Classe #859 que é "toaster"(torradeira) abaixo
object_type_to_fake = 859

# Carrega o arquivo da imagem original e o converte em um array para o numpy utilizar
# essa imagem carregada de um gato será passada como se fosse de uma torradeira
img = image.load_img("cat.png", target_size=(299, 299))
original_image = image.img_to_array(img)

# Redimensiona a imagem para que a intensidade de todos os pixels
# varie entre [-1, 1] como o modelo espera receber o objeto
original_image /= 255.
original_image -= 0.5
original_image *= 2.

# Adiciona uma quarta dimensão para o tamanho desse lote,
# como Keras espera receber a imagem assim, é enviado aqui
original_image = np.expand_dims(original_image, axis=0)

# Pré Calcula a Mudança Máxima que permitiremos na imagem (de informações, dimensões e nitidez)
# Usa-se isso para se ter certeza que a imagem não sairá ruim ou inválida,
# a deixa plausível como uma imagem digital ou até física (impressa) caso precise
# Um número mais largo produz uma imagem mais rápido,
# mas o faz correndo risco de haver mais distorção na imagem
max_change_above = original_image + 0.01
max_change_below = original_image - 0.01

# Cria uma cópia da imagem do gato para ser transformada
changed_image = np.copy(original_image)

# Quanto da imagem será alterada a cada iteração atráves do loop da rede neural
learning_rate = 0.1

# Definindo a função de custo
# O 'custo' será a chance da imagem ser o que definimos em sua classe
# de acordo com a interpretação do Modelo com os dados de pré-treinamento
cost_function = model_output_layer[0, object_type_to_fake]

# Aqui é calculado o gradiente com ajuda da Keras, baseando-se na imagem original
# e na Classe que foi prevista nesse momento
# Nesse caso, nos referindo ao "model_input_layer", é
# retornada a imagem original
gradient_function = K.gradients(cost_function, model_input_layer)[0]


# Função do Keras que podemos chamar para calcular a chance (da Classe)
# e o gradiente, ambos sendo atuais
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

cost = 0.0

# Em um loop, a imagem transformada continua sendo alterada mais e mais até
# Ser passada pelo modelo com no mínimo 80% de credibilidade
while cost < 0.80:
    # Verifica o quão próximo nossa imagem está da Classe que escolhemos anteriormente
    # Nesse caso a Torradeira, e pega os gradientes para que possam ser transformados
    # Para avançarem um passo a mais a cada iteração da Classe Torradeira que foi definida
    # Nota: É muito importante passar o '0' para que o modelo de treinamento Keras aqui.
    # As camadas (layers) da Keras se comportam diferente em predições vs modelos de treinamento.
    cost, gradients = grab_cost_and_gradients_from_model([changed_image, 0])

    # Move a imagem transformada um passo a mais para se passar pela classe
    # Torradeira neste modelo de treinamento
    changed_image += gradients * learning_rate

    # Isso assegura que a imagem não mude muito para ficar ruim (implausível) ou inválida (totalmente fora de contexto)
    changed_image = np.clip(changed_image, max_change_below, max_change_above)
    changed_image = np.clip(changed_image, -1.0, 1.0)

    print("O modelo previu que a chance da imagem ser uma torradeira é: {:.8}%".format(cost * 100))

# Depois da otimização, retorna a escala dos pixels da imagem de [-1, 1] de volta ao
# ao padrão de [0, 255] (range) nessa imagem transformada
img = changed_image[0]
img /= 2.
img += 0.5
img *= 255.

# Salva essa imagem transformada
im = Image.fromarray(img.astype(np.uint8))
im.save("changed-image.png")
