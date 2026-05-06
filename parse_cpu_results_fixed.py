import os
import re
import pandas as pd

def parse_log_file(file_path):
    algorithm = os.path.basename(file_path).split('_')[0]
    dataset = os.path.basename(file_path).split('_')[1].replace('.log', '')
    
    distances = []
    runtimes = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if 'Total Distance' in line:
            match = re.search(r'Total Distance:?\s*([\d.]+)', line)
            if match:
                dist = float(match.group(1))
                distances.append(dist)
        elif '运行时间' in line:
            match = re.search(r'运行时间:?\s*([\d.]+)\s*秒', line)
            if match:
                runtime = float(match.group(1))
                runtimes.append(runtime)
    
    if not distances or not runtimes:
        print(f"⚠️  {file_path} 数据不完整")
        return None
    
    return {
        '算法': algorithm,
        '数据集': dataset,
        '平均距离': round(sum(distances)/len(distances), 2),
        '最小距离': round(min(distances), 2),
        '平均运行时间(秒)': round(sum(runtimes)/len(runtimes), 2),
        '标准差(秒)': round(pd.Series(runtimes).std(), 2) if len(runtimes) > 1 else 0,
        '运行次数': len(runtimes)
    }

if __name__ == "__main__":
    results_dir = 'results/cpu'
    all_results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(results_dir, filename)
            result = parse_log_file(file_path)
            if result:
                all_results.append(result)
    
    df = pd.DataFrame(all_results)
    df = df.sort_values(['算法', '数据集'])
    
    print("="*100)
    print("📊 result for CPU Baseline Testing")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    df.to_csv('cpu_baseline_results.csv', index=False, encoding='utf-8-sig')
    print("✅ save as cpu_baseline_results.csv")