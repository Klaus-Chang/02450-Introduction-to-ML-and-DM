import numpy as np

# 假设您的OCR数据存储在一个字符串变量中
ocr_data = """
0 0.534 1.257 1.671 1.090 1.315 1.484 1.253 1.418
0.534 0 0.727 2.119 1.526 1.689 1.214 0.997 1.056
1.257 0.727 0 2.809 2.220 2.342 1.088 0.965 0.807
1.671 2.119 2.809 0 0.601 0.540 3.135 2.908 3.087
1.090 1.526 2.220 0.601 0 0.331 2.563 2.338 2.500
1.315 1.689 2.342 0.540 0.331 0 2.797 2.567 2.708
1.484 1.214 1.088 3.135 2.563 2.797 0 0.275 0.298
1.253 0.997 0.965 2.908 2.338 2.567 0.275 0 0.343
1.418 1.056 0.807 3.087 2.500 2.708 0.298 0.343 0
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
    # 获取数组的字符串表示形式，逐行处理
    str_arr = str(arr).replace('[', '').replace(']', '').split('\n')
    # 打印格式化后的数组
    print('[', end='')
    for i, line in enumerate(str_arr):
        # 处理每一行，添加逗号和换行符
        formatted_line = "['" + "', '".join(line.split()) + "'],"
        if i < len(str_arr) - 1:
            print(formatted_line)
        else:
            # 最后一行不需要逗号
            print(formatted_line[:-1])
    print(']')

# 使用自定义函数打印数组
pretty_print_numpy_array(distance_matrix)
