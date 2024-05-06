#Descobridor do Valor das Casas <(DVC)>
#import ast
#Pode processar os dados, nesse caso prefiri mais comedimento
#A pesquisa desse método foi por agnição, os preços estão em dólares ($)

def Casas():
    print("Bem-vindo(a) ao Descobridor de Valor!")
    print("Digite as informações da casa, uma por vez.")
    num_de_quartos = int(input("Digite o número de quartos: "))
    largura = int(input("Digite a largura em metros: "))
    altura = int(input("Digite a altura em metros: "))
    dimensoes = (largura*altura)*10000
    vizinhanca = int(input("Digite o número de casas ao redor: "))
    valor = Estimativa_do_preco_de_casas(num_de_quartos, dimensoes, vizinhanca)
    retorna = "{0:.2f}".format(valor)
    
    print("O preço dessa casa é: " + retorna + "$.")

def Estimativa_do_preco_de_casas(num_de_quartos, dimensoes, vizinhanca):
# Essa é uma função para Inteligência Artificial
 preco = 0
# Uma pitada disso
 preco += num_de_quartos * 0.123
# Uma pitada melhor disso
 preco += dimensoes * 0.41
# Talvez uma pitada disso
 preco += vizinhanca * 0.57
# E Vo-a-lá!
 return preco

Casas()

