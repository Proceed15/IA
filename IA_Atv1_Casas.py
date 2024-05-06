#Descobrdor de Valor de Casas <(DVC)>
#import ast
#Pode processar os dados, nesse caso prefiri mais comedimento.
#A pesquisa desse método foi por agnição.

def Casas():
    print("Bem-vindo(a) ao Descobridor de Valor!")
    print("Digite as informações da casa, uma por vez.")
    num_de_quartos = int(input("Digite o número de quartos: "))
    largura = int(input("Digite a largura: "))
    altura = int(input("Digite a altura: "))
    dimensoes = largura*altura
    vizinhanca = int(input("digite o número de casas ao redor: "))
    valor = Estimativa_do_preco_de_casas(num_de_quartos, dimensoes, vizinhanca)
    
    print("O preço da residência é: " + str(valor) + "$.")

def Estimativa_do_preco_de_casas(num_de_quartos, dimensoes, vizinhanca):
 preco = 0
# Uma pitada disso
 preco += num_de_quartos * 0.123
# Uma melhor pitada disso
 preco += dimensoes * 0.41
# Talvez uma pitada disso
 preco += vizinhanca * 0.57
# E Vo-a-lá!
 return preco

Casas()

