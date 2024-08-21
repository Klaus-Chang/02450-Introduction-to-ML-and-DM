import numpy as np
from scipy.stats import skew

def analyze_distribution(percentiles):
    """
    分析数据分布，计算 99%-1% 差值、全距、偏度和标准差。
    
    参数：
    percentiles: dict 包含1st, 50th, 99th, min, max百分位数值
    
    返回：
    分析结果的字典，包括 range_99_1, data_range, skewness, std_dev
    """
    # 计算 99%-1% 的差值
    range_99_1 = percentiles['99th'] - percentiles['1st']
    
    # 计算全距（使用给定的最大值和最小值）
    data_range = percentiles['max'] - percentiles['min']
    
    # 模拟数据来计算偏度和标准差
    # 这里假设数据分布较大范围为正态分布，实际数据分析时应基于实际数据
    simulated_data = np.random.normal(loc=percentiles['50th'], scale=1, size=1000)
    
    # 计算偏度
    skewness = skew(simulated_data)
    
    # 计算标准差
    std_dev = np.std(simulated_data)
    
    return {
        'range_99_1': range_99_1,
        'data_range': data_range,
        'skewness': skewness,
        'std_dev': std_dev
    }

def main():
    # 输入题目信息
    percentiles_x1 = {'1st': -1.19, '50th': -0.08, '99th': 1.21, 'min': -1.19, 'max': 1.21}
    percentiles_x2 = {'1st': -1.6, '50th': -0.13, '99th': 2.59, 'min': -1.6, 'max': 2.59}
    percentiles_x3 = {'1st': -1.76, '50th': 0.03, '99th': 1.48, 'min': -1.76, 'max': 1.48}
    percentiles_x4 = {'1st': -0.7, '50th': 0.31, '99th': 3.21, 'min': -0.7, 'max': 3.21}
    percentiles_x5 = {'1st': -1.21, '50th': -0.23, '99th': 2.25, 'min': -1.21, 'max': 2.25}
    
    # 分析每个属性的数据分布
    analysis_x1 = analyze_distribution(percentiles_x1)
    analysis_x2 = analyze_distribution(percentiles_x2)
    analysis_x3 = analyze_distribution(percentiles_x3)
    analysis_x4 = analyze_distribution(percentiles_x4)
    analysis_x5 = analyze_distribution(percentiles_x5)
    
    # 输出分析结果
    print("Analysis for x1:", analysis_x1)
    print("Analysis for x2:", analysis_x2)
    print("Analysis for x3:", analysis_x3)
    print("Analysis for x4:", analysis_x4)
    print("Analysis for x5:", analysis_x5)
    
    # 判断和直方图的匹配
    if analysis_x4['range_99_1'] > analysis_x1['range_99_1'] and analysis_x4['skewness'] > 0:
        print("x4 数据分布更广且右偏，可能对应图 B")
    if analysis_x1['range_99_1'] < analysis_x4['range_99_1'] and abs(analysis_x1['skewness']) < 0.5:
        print("x1 数据较集中且接近对称，可能对应图 A")

if __name__ == "__main__":
    main()
