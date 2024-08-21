import numpy as np

# 假设您的OCR数据存储在一个字符串变量中
ocr_data = """
0.6667 0.1524 0.1143 0.1714 0.1714 0.1619 0.1619 0.1619 0.1333 0.1429 0.1619 0.1714 0.1619 0.1619 0.1524 0.1810
"""

# 将字符串数据按行分割
lines = ocr_data.strip().split('\n')

# 提取每行的数字并转换为浮点数
matrix_data = []
for line in lines:
    # 分割每行的数字
    numbers = line.split()
    # 将字符串转换为浮点数
    float_numbers = [float(num) for num in numbers]
    matrix_data.append(float_numbers)

# 将数据转换为NumPy数组
distance_matrix = np.array(matrix_data)

# 定义一个函数来美化打印NumPy数组
def pretty_print_numpy_array(arr):
    # 获取数组的字符串表示形式
    str_arr = str(arr).replace('[', '').replace(']', '').replace('\n', '')
    # 打印格式化后的数组
    print('[' + ', '.join(str_arr.split()) + ']')


# 使用自定义函数打印数组
pretty_print_numpy_array(distance_matrix)
