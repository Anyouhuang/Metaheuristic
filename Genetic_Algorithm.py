# -*- coding: utf-8 -*-
import random
import numpy as np
import math

# --- 參數設定 ---
POPULATION_SIZE = 50
X_MIN, X_MAX = -5.12, 5.12
PC = 0.8  # Crossover probability
PM = 0.1  # Mutation probability
NUM_ITERATIONS = 10000
NUM_RUNS = 10

def calculate_gene_length(epsilon, x_min, x_max):
    """
    根據所需精確度 epsilon 計算基因長度 n。
    
    公式: n >= log2( (x_max - x_min) / epsilon + 1 )
    """
    length = (x_max - x_min) / epsilon
    n = math.ceil(math.log2(length + 1))
    return int(n)

# --- 核心函式 ---
def decode_gene(gene, gene_length, x_min, x_max):
    """將基因解碼成x值"""
    x = []
    total_gene_length = 3 * gene_length
    for i in range(3):
        sub_gene = gene[i * gene_length : (i + 1) * gene_length]
        decimal_val = int("".join(map(str, sub_gene)), 2)
        decoded_val = x_min + (decimal_val / (2**gene_length - 1)) * (x_max - x_min)
        x.append(decoded_val)
    return x

def objective_func(x_list):
    """目標函數 f(x) = sum(x_i^2)"""
    return sum(xi**2 for xi in x_list)

def fitness_func(x_list):
    """適應度函數 (尋找最小值，所以用倒數)"""
    return 1 / (objective_func(x_list) + 1e-6)

def single_point_crossover(parent1, parent2):
    """單點交配"""
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def two_point_mutation(gene, total_gene_length):
    """兩點突變"""
    gene = list(gene)  # 轉成list方便修改
    p1, p2 = random.sample(range(total_gene_length), 2)
    gene[p1] = 1 - gene[p1]
    gene[p2] = 1 - gene[p2]
    return tuple(gene)

def run_ga(gene_length):
    """執行一次遺傳演算法"""
    total_gene_length = 3 * gene_length
    
    # 初始化族群
    population = [tuple(random.randint(0, 1) for _ in range(total_gene_length)) for _ in range(POPULATION_SIZE)]

    best_solution = float('inf')
    best_chromosome = None

    for _ in range(NUM_ITERATIONS):
        fitness_scores = [fitness_func(decode_gene(p, gene_length, X_MIN, X_MAX)) for p in population]
        
        best_idx = np.argmax(fitness_scores)
        current_best_x = decode_gene(population[best_idx], gene_length, X_MIN, X_MAX)
        current_best_val = objective_func(current_best_x)

        if current_best_val < best_solution:
            best_solution = current_best_val
            best_chromosome = population[best_idx]
            
        elites = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        new_population = [p for p, f in elites[:2]]

        while len(new_population) < POPULATION_SIZE:
            total_fitness = sum(fitness_scores)
            probabilities = [f / total_fitness for f in fitness_scores]
            parent1 = random.choices(population, weights=probabilities, k=1)[0]
            parent2 = random.choices(population, weights=probabilities, k=1)[0]

            child1, child2 = parent1, parent2
            if random.random() < PC:
                child1, child2 = single_point_crossover(parent1, parent2)

            if random.random() < PM:
                child1 = two_point_mutation(child1, total_gene_length)
            if random.random() < PM:
                child2 = two_point_mutation(child2, total_gene_length)
            
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

    return best_solution, best_chromosome

# --- 主程式碼 (執行多次) ---

if __name__ == "__main__":
    # 設定精確度10^-3
    desired_precision = 0.001
    
    # 根據精確度計算基因長度
    gene_length_n = calculate_gene_length(desired_precision, X_MIN, X_MAX)
    total_gene_length_n = 3 * gene_length_n

    print("="*40)
    print(f"根據期望精確度 ε = {desired_precision}")
    print(f"每個變數所需的最小基因長度 (n) = {gene_length_n} 位元")
    print(f"總基因長度 = {total_gene_length_n} 位元")
    print("="*40 + "\n")

    results = []
    for i in range(NUM_RUNS):
        print(f"--- 執行第 {i+1} 次 ---")
        best_val, best_chrom = run_ga(gene_length_n)
        results.append(best_val)
        print(f"  本次最佳解的函數值: {best_val:.8f}")
        print(f"  本次最佳解的x值: {[round(x, 4) for x in decode_gene(best_chrom, gene_length_n, X_MIN, X_MAX)]}")
        print("-" * 20)

    # 統計所有結果
    results = np.array(results)
    print("\n" + "="*30)
    print("      所有實驗的統計結果      ")
    print("="*30)
    print(f"平均最佳解值: {results.mean():.8f}")
    print(f"標準差: {results.std():.8f}")
    print(f"最佳結果 (最小值): {results.min():.8f}")
    print(f"最差結果 (最大值): {results.max():.8f}")
    print("="*30)