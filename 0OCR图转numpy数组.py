import numpy as np

# 假设您的OCR数据存储在一个字符串变量中
ocr_data = """

0.0 10.4 11.3 3.5 10.3 6.3 5.1 8.6 10.8 11.0
10.4 0.0 5.5 7.1 0.4 4.1 5.3 1.8 4.4 6.1
11.3 5.5 0.0 8.0 5.8 9.0 10.2 6.7 4.6 6.4
3.5 7.1 8.0 0.0 7.0 3.7 2.7 5.4 7.5 8.2
10.3 0.4 5.8 7.0 0.0 4.0 5.2 1.7 4.5 5.9
6.3 4.1 9.0 3.7 4.0 0.0 1.7 2.3 8.5 9.1
5.1 5.3 10.2 2.7 5.2 1.7 0.0 3.5 9.7 9.9
8.6 1.8 6.7 5.4 1.7 2.3 3.5 0.0 6.2 6.8
10.8 4.4 4.6 7.5 4.5 8.5 9.7 6.2 0.0 2.5
11.0 6.1 6.4 8.2 5.9 9.1 9.9 6.8 2.5 0.0

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
        formatted_line = '[' + ', '.join(line.split()) + '],'
        if i < len(str_arr) - 1:
            print(formatted_line)
        else:
            # 最后一行不需要逗号
            print(formatted_line[:-1])
    print(']')

# 使用自定义函数打印数组
pretty_print_numpy_array(distance_matrix)