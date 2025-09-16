import numpy as np

# 目標函數 f(x)
def objective(x):
    return np.sum(x**2)

# 模擬退火演算法
def simulated_annealing(max_iter=1000, T0=5, alpha=0.95, sigma=0.5):
    # 初始解
    x = np.random.uniform(-5.12, 5.12, 3)
    fx = objective(x)
    
    best_x, best_fx = x.copy(), fx
    T = T0
    
    for k in range(max_iter):
        # 產生新解
        x_new = x + np.random.normal(0, sigma, size = 3)
        # print(x_new-x)
        # 保持在範圍內
        x_new = np.clip(x_new, -5.12, 5.12)
        fx_new = objective(x_new)
        
        # 判斷是否接受
        if fx_new < fx:
            x, fx = x_new, fx_new
        else:
            # 機率接受
            if np.random.rand() < np.exp(-(fx_new - fx) / T):
                x, fx = x_new, fx_new
        
        # 更新最佳解
        if fx < best_fx:
            best_x, best_fx = x.copy(), fx
        
        # 降溫
        T *= alpha
    
    return best_x, best_fx

# 多次重複運行
results = []
for i in range(10):
    _, fx = simulated_annealing()
    results.append(fx)

# 統計結果
results = np.array(results)
print("10 runs results:", results)
print("平均值:", results.mean())
print("標準差:", results.std())
print("最佳解:", results.min())
print("最差解:", results.max())
