import copy
import random
import numpy as np

def gerar_vizinhos_knapsack(solucao, n_vizinhos=10):
    """
    Gera vizinhos para o problema knapsack
    Estratégia: flip de um bit aleatório

    Args:
        solucao: solução binária atual
        n_vizinhos: número de vizinhos

    Returns:
        list: lista de vizinhos
    """
    vizinhos = []
    n_itens = len(solucao)

    # Gerar vizinhos por flip de bit
    sorted_pos = []
    for i in range(n_vizinhos):
        # Escolher posição aleatória para flip
        pos = random.randint(0, n_itens - 1)
        if pos in sorted_pos:
            continue

        vizinho = solucao.copy()
        vizinho[pos] = 1 - vizinho[pos]  # Flip do bit
        vizinhos.append(vizinho)
        sorted_pos.append(pos)

    return vizinhos

class HillClimbing:
    def __init__(self, funcao_fitness, gerar_vizinhos, maximizar=True):
        """
        Inicializa o algoritmo Hill Climbing

        Args:
            funcao_fitness: função que avalia soluções
            gerar_vizinhos: função que gera vizinhos de uma solução
            maximizar: True para maximização, False para minimização
        """
        self.funcao_fitness = funcao_fitness
        self.gerar_vizinhos = gerar_vizinhos
        self.maximizar = maximizar
        self.historico = []

    def executar(self, solucao_inicial, max_iteracoes=1000, verbose=False, estocastico=False, probabilidade_pior=0.1):
        """
        Executa o algoritmo Hill Climbing

        Args:
            solucao_inicial: solução inicial
            max_iteracoes: número máximo de iterações
            verbose: imprimir progresso
            estocastico: True para usar a versão estocástica
            probabilidade_pior: probabilidade de aceitar uma solução pior no modo estocástico

        Returns:
            tuple: (melhor_solucao, melhor_fitness, historico)
        """
        solucao_atual = copy.deepcopy(solucao_inicial)
        fitness_atual = self.funcao_fitness(solucao_atual)

        self.historico = [fitness_atual]
        iteracao = 0
        melhorias = 0

        if verbose:
            print(f"Iteração {iteracao}: Fitness = {fitness_atual:.4f}")

        while iteracao < max_iteracoes:
            iteracao += 1

            # Gerar vizinhos
            vizinhos = self.gerar_vizinhos(solucao_atual)

            # Avaliar vizinhos e encontrar o melhor
            melhor_vizinho = None
            melhor_fitness_vizinho = fitness_atual

            for vizinho in vizinhos:
                fitness_vizinho = self.funcao_fitness(vizinho)

                # Verificar se é melhor
                eh_melhor = (
                    fitness_vizinho > melhor_fitness_vizinho
                    if self.maximizar
                    else fitness_vizinho < melhor_fitness_vizinho
                )

                if eh_melhor:
                    melhor_vizinho = vizinho
                    melhor_fitness_vizinho = fitness_vizinho
                elif estocastico and random.random() < probabilidade_pior:
                    # Na versão estocástica, podemos aceitar uma solução pior
                    melhor_vizinho = vizinho
                    melhor_fitness_vizinho = fitness_vizinho
                    break  # Aceita o primeiro vizinho "pior" que passar no teste de probabilidade


            # Se encontrou vizinho melhor (ou um pior aceito no modo estocástico), move para ele
            if melhor_vizinho is not None:
                solucao_atual = copy.deepcopy(melhor_vizinho)
                fitness_atual = melhor_fitness_vizinho
                melhorias += 1

                if verbose:
                    print(f"Iteração {iteracao}: Fitness = {fitness_atual:.4f}")
            else:
                # Nenhum vizinho melhor encontrado - parar
                if verbose:
                    print(f"Convergiu na iteração {iteracao}")
                break

            self.historico.append(fitness_atual)

        if verbose:
            print(f"Melhorias realizadas: {melhorias}")
            print(f"Fitness final: {fitness_atual:.4f}")

        return solucao_atual, fitness_atual, self.historico

if __name__ == "__main__":
    import sys
    from knapsack import knapsack
    import random
    import numpy as np

    # Configuração do problema knapsack
    DIM = 20
    MAX_ITERACOES = 200

    # --- Execução do Hill Climbing Padrão ---
    print("\n=== EXECUTANDO HILL CLIMBING PADRÃO ===")
    melhores_fitness_padrao = []
    for i in range(30):
        # Gerar solução inicial aleatória
        solucao_inicial = [int(random.random() > 0.8) for _ in range(DIM)]
        
        hill_climbing = HillClimbing(    
            funcao_fitness=lambda sol: knapsack(sol, dim=DIM)[0],  # Maximizar valor total
            gerar_vizinhos=gerar_vizinhos_knapsack,
            maximizar=True,
        )

        _, melhor_fitness, _ = hill_climbing.executar(
            solucao_inicial, max_iteracoes=MAX_ITERACOES, verbose=False # verbose=False para não poluir o output
        )
        melhores_fitness_padrao.append(melhor_fitness)

    print("Resultados Hill Climbing Padrão:", melhores_fitness_padrao)
    np.savetxt("resultados_finais.txt", melhores_fitness_padrao, fmt='%d')
    print("Média (Padrão):", np.mean(melhores_fitness_padrao))
    print("Desvio Padrão (Padrão):", np.std(melhores_fitness_padrao))

    # --- Execução do Hill Climbing Estocástico ---
    print("\n=== EXECUTANDO HILL CLIMBING ESTOCÁSTICO ===")
    melhores_fitness_estocastico = []
    for i in range(30):
        # Gerar solução inicial aleatória
        solucao_inicial = [int(random.random() > 0.8) for _ in range(DIM)]
        
        hill_climbing_estocastico = HillClimbing(    
            funcao_fitness=lambda sol: knapsack(sol, dim=DIM)[0],  # Maximizar valor total
            gerar_vizinhos=gerar_vizinhos_knapsack,
            maximizar=True,
        )

        _, melhor_fitness, _ = hill_climbing_estocastico.executar(
            solucao_inicial, max_iteracoes=MAX_ITERACOES, verbose=False, estocastico=True, probabilidade_pior=0.05
        )
        melhores_fitness_estocastico.append(melhor_fitness)

    print("Resultados Hill Climbing Estocástico:", melhores_fitness_estocastico)
    np.savetxt("resultados_finais2.txt", melhores_fitness_estocastico, fmt='%d')
    print("Média (Estocástico):", np.mean(melhores_fitness_estocastico))
    print("Desvio Padrão (Estocástico):", np.std(melhores_fitness_estocastico))

    #print p comparar

    print("\n\n" + "="*50)
    print("=== COMPARAÇÃO DOS ALGORITMOS ===")
    print("="*50)
    print(f"{'Métrica':<20} | {'Hill Climbing Padrão':^20} | {'Hill Climbing Estocástico':^20}")
    print("-"*50)

    media_padrao = np.mean(melhores_fitness_padrao)
    std_padrao = np.std(melhores_fitness_padrao)
    max_padrao = np.max(melhores_fitness_padrao)

    media_estocastico = np.mean(melhores_fitness_estocastico)
    std_estocastico = np.std(melhores_fitness_estocastico)
    max_estocastico = np.max(melhores_fitness_estocastico)

    print(f"{'Média do Fitness':<20} | {media_padrao:^20.2f} | {media_estocastico:^20.2f}")
    print(f"{'Desvio Padrão':<20} | {std_padrao:^20.2f} | {std_estocastico:^20.2f}")
    print(f"{'Melhor Resultado':<20} | {max_padrao:^20} | {max_estocastico:^20}")
    print("="*50)