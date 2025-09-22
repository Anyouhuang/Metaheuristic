import numpy as np
import random
import math

# --- 1. 定義城市距離矩陣 ---
# 根據提供的 PDF 文件，將 12 個城市的距離數據轉換為 NumPy 陣列。
# 城市索引從 0 開始，對應到 PDF 中的城市 1 到 12。
# 城市 1 是起點和終點 (索引為 0)。
distance_matrix = np.array([
    [0, 112, 49, 53, 13, 80, 12, 40, 78, 71, 72, 12],
    [112, 0, 71, 81, 122, 83, 106, 82, 91, 67, 40, 117],
    [49, 71, 0, 16, 62, 81, 39, 46, 42, 24, 34, 50],
    [53, 81, 16, 0, 66, 96, 41, 60, 28, 20, 47, 51],
    [13, 122, 62, 66, 0, 81, 25, 45, 91, 84, 83, 20],
    [80, 83, 81, 96, 81, 0, 85, 41, 122, 101, 64, 91],
    [12, 106, 39, 41, 25, 85, 0, 43, 65, 60, 66, 11],
    [40, 82, 46, 60, 45, 41, 43, 0, 87, 69, 47, 51],
    [78, 91, 42, 28, 91, 122, 65, 87, 0, 25, 66, 73],
    [71, 67, 24, 20, 84, 101, 60, 69, 25, 0, 41, 70],
    [72, 40, 34, 47, 83, 64, 66, 47, 66, 41, 0, 78],
    [12, 117, 50, 51, 20, 91, 11, 51, 73, 70, 78, 0]
])
# 城市數量
num_cities = len(distance_matrix)

# --- 2. 輔助函數 ---

def calculate_total_distance(path):
    """計算給定路徑的總距離。路徑必須以城市 1 (索引 0) 為起點和終點。"""
    total_dist = 0
    for i in range(len(path) - 1):
        from_city = path[i]
        to_city = path[i+1]
        total_dist += distance_matrix[from_city][to_city]
    # 加上從最後一個城市回到起點城市 1 的距離
    total_dist += distance_matrix[path[-1]][path[0]]
    return total_dist

def nearest_neighbor_path(start_city):
    """使用最近鄰演算法從一個給定的城市產生一個路徑。"""
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)
    current_city = start_city
    path = [current_city]

    while unvisited:
        nearest_city = -1
        min_dist = float('inf')
        for neighbor in unvisited:
            dist = distance_matrix[current_city][neighbor]
            if dist < min_dist:
                min_dist = dist
                nearest_city = neighbor
        
        path.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
    
    # 移除最後一個城市，因為我們將在主函數中處理起點和終點的關係
    return path

def generate_initial_solution():
    """
    根據要求生成初始解。
    方法：從 5 個隨機選擇的城市開始，使用最近鄰演算法，然後選擇其中總距離最短的那個解。
    """
    best_path = None
    min_distance = float('inf')
    
    # 隨機選擇 5 個起點 (城市 1 以外的城市)
    # 城市 1 (索引 0) 是固定的起點，但我們可以從其他城市開始生成路徑
    random_starts = random.sample(range(1, num_cities), 5)
    
    for start_city in random_starts:
        path_without_start = nearest_neighbor_path(start_city)
        # 將城市 1 (索引 0) 插入路徑的開頭
        current_path = [0] + path_without_start
        current_distance = calculate_total_distance(current_path)
        
        if current_distance < min_distance:
            min_distance = current_distance
            best_path = current_path
            
    # 如果 5 次隨機開始都沒有找到解，則從城市 1 開始
    if best_path is None:
        best_path = nearest_neighbor_path(0)
    
    return best_path, min_distance


def two_opt_swap(path, i, j):
    """
    執行 2-opt 交換操作，翻轉路徑中 i 和 j 之間的片段。
    這將產生一個新的鄰居路徑。
    """
    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
    return new_path

# --- 3. 禁忌搜尋演算法主函數 ---
def tabu_search_tsp():
    """
    禁忌搜尋演算法主函數。
    """
    # 演算法參數
    t_max = 10000  # 最大迭代次數
    tabu_tenure = 7  # 禁忌期限
    
    # 初始化禁忌清單 (使用字典來儲存，鍵是禁忌屬性，值是解禁的迭代次數)
    tabu_list = {}
    
    # 產生初始解
    current_path, current_distance = generate_initial_solution()
    
    # 確保城市 1 (索引 0) 在路徑的開頭，這是固定的。
    if current_path[0] != 0:
        current_path.remove(0)
        current_path.insert(0, 0)
    
    best_path = current_path[:]
    best_distance = current_distance
    
    # 禁忌搜尋主迴圈
    for t in range(t_max):
        
        # 尋找最佳鄰居
        best_neighbor_path = None
        best_neighbor_distance = float('inf')
        best_move_attribute = None
        
        # 產生所有 2-opt 鄰居
        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                # 執行 2-opt 交換
                neighbor_path = two_opt_swap(current_path, i, j)
                neighbor_distance = calculate_total_distance(neighbor_path)
                
                # 定義禁忌屬性：移除的兩條邊 (i, i+1) 和 (j, j+1)
                move_attribute = frozenset({(current_path[i-1], current_path[i]), (current_path[j], current_path[j+1])})
                
                # 檢查是否為禁忌移動
                is_tabu = False
                if move_attribute in tabu_list and tabu_list[move_attribute] > t:
                    is_tabu = True
                
                # 渴望準則：允許禁忌移動如果它能產生比歷史最佳解更好的解
                aspiration_criteron_met = False
                if is_tabu and neighbor_distance < best_distance:
                    aspiration_criteron_met = True
                
                # 如果不是禁忌移動，或是滿足渴望準則，則考慮這個移動
                if not is_tabu or aspiration_criteron_met:
                    if neighbor_distance < best_neighbor_distance:
                        best_neighbor_distance = neighbor_distance
                        best_neighbor_path = neighbor_path
                        best_move_attribute = move_attribute
        
        # 如果沒有找到可行的鄰居，則跳出迴圈
        if best_neighbor_path is None:
            break
            
        # 更新當前解
        current_path = best_neighbor_path
        current_distance = best_neighbor_distance
        
        # 更新禁忌清單
        if best_move_attribute is not None:
            tabu_list[best_move_attribute] = t + tabu_tenure
        
        # 更新歷史最佳解
        if current_distance < best_distance:
            best_distance = current_distance
            best_path = current_path[:]
            
    return best_distance, best_path

# --- 4. 執行多次並計算統計數據 ---
def run_and_analyze_tabu_search():
    """
    執行禁忌搜尋 10 次，並計算統計數據。
    """
    run_results = []
    print("--- 執行禁忌搜尋 10 次 ---")
    for i in range(10):
        final_distance, final_path = tabu_search_tsp()
        run_results.append(final_distance)
        print(f"第 {i+1} 次執行: 最佳路徑長度 = {final_distance}")
    
    print("\n--- 統計結果 ---")
    results_array = np.array(run_results)
    
    # 計算平均值
    average = np.mean(results_array)
    # 計算標準差
    std_dev = np.std(results_array)
    # 找到最佳解
    best_solution = np.min(results_array)
    # 找到最差解
    worst_solution = np.max(results_array)
    
    print(f"平均值: {average:.2f}")
    print(f"標準差: {std_dev:.2f}")
    print(f"最佳解: {best_solution}")
    print(f"最差解: {worst_solution}")

# 運行主程式
if __name__ == "__main__":
    run_and_analyze_tabu_search()
