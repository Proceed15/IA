# Código Adicional para calcular o espaço parcial ocupado em uma vaga
# Usando esse codigo no algoritmo principal, é possível calcular a
# Intersection Over Unit (IoU) de cada vaga

# Também é possível usar a função mrcnn.utils.compute_overlaps()
# Do Recurso Matterport Mask R-CNN (library)

 # Filtro dos resultados para pegar somente as bordas das caixas de carros
 # e caminhões, ou seja, onde há a vaga
car_boxes = get_car_boxes(r['rois'], r['class_ids'])

 # Descubra quanto espaço os carros ocupam sobre vagas conhecidas
overlaps = mrcnn.utils.compute_overlaps(car_boxes, parking_areas)

 # Mostra o espaço ocupado
print(overlaps)

""" O resultado, quando executado ao programa principal deve ser algo assim:
[
 [1.         0.07040032 0.         0.]
 [0.07040032 1.         0.07673165 0.]
 [0.         0.         0.02332112 0.]
]
"""
#Cada componente na linha retornada é o espaço ocupado
#1. Seria um espaço totalmente ocupado
#0.02332112 Seria um espaço praticamente inocupado
#Lembre-se de integrar o código e seus resultados com as filmagens reais
#Para garantir a integridade dos resultados, garanta que o espaço ocupado
#Esteja semelhante durante os trechos de vídeo durante 5 a 10 segundos ou mais
#Faça isso comparando os resultados dos espaçamentos em diferentes tempos.
