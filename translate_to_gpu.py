from openai import OpenAI

# 替换成你自己的DeepSeek API Key
DEEPSEEK_API_KEY = "sk-16bf633f946f4e56ac7487035b76af53"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def translate_algorithm(algorithm_name):
    # 读取本地的CPU代码
    with open(f"{algorithm_name}.py", "r") as f:
        cpu_code = f.read()
    
    # 针对A30 GPU优化的硬件感知提示词
    prompt = f"""
    请将下面这个Python实现的ACO CVRP算法转换为**高度优化的NVIDIA A30 GPU CUDA C++代码**。

    严格遵守以下要求：
    1. 100%算法正确性：输出的总距离必须与原Python代码完全一致
    2. 针对ACO的并行优化：
    - 每只蚂蚁分配一个GPU线程
    - 信息素更新在GPU上并行完成
    - 使用共享内存缓存距离矩阵和信息素矩阵
    3. 输入输出：
    - 接受CVRPLib .txt文件路径作为唯一命令行参数
    - 输出格式：先打印"Total Distance: X.X"
    4. 代码质量：
    - 添加完整的CUDA错误检查
    - 所有动态分配的内存都要正确释放
    - 可直接用nvcc 12.4编译运行

    原Python代码：
    [粘贴aco_cvrp.py的完整内容]
    {cpu_code}
    """
    
    print(f"🔄 正在翻译 {algorithm_name} 算法到GPU代码...")
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=8192
    )
    
    # 提取CUDA代码（自动去掉代码块标记）
    gpu_code = response.choices[0].message.content
    if "```cuda" in gpu_code:
        gpu_code = gpu_code.split("```cuda")[1].split("```")[0]
    elif "```cpp" in gpu_code:
        gpu_code = gpu_code.split("```cpp")[1].split("```")[0]
    
    # 保存为CUDA文件
    output_file = f"{algorithm_name}_gpu.cu"
    with open(output_file, "w") as f:
        f.write(gpu_code)
    
    print(f"✅ {algorithm_name} GPU代码已生成并保存为 {output_file}")
    return output_file

if __name__ == "__main__":
    translate_algorithm("aco_cvrp")