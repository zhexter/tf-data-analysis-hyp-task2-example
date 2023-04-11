import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 123456 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    p_value = MMD(compute_kernel="laplacian", gamma=1).test(x, y)[1]
    alpha = 0.06
    return p_value < alpha # Ваш ответ, True или False