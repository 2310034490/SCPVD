import pandas as pd
import json
import os


def convert_csv_to_json(csv_file_path, json_file_path):
    """
    将BigVul CSV数据集转换为JSON格式
    只保留漏洞标识和代码片段两个字段，重命名为target和func
    """
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        print(f"原始数据集大小: {len(df)}")
        print(f"原始列名: {list(df.columns)}")
        
        # 检查必要的列是否存在
        # BigVul数据集列名可能是: 'vul'表示漏洞
        # 'func_before' 表示代码
        vulnerability_cols = ['vul']
        code_cols = ['func_before']
        
        vul_col = None
        code_col = None
        
        # 查找漏洞标识列
        for col in vulnerability_cols:
            if col in df.columns:
                vul_col = col
                break
        
        # 查找代码列
        for col in code_cols:
            if col in df.columns:
                code_col = col
                break
        
        if vul_col is None:
            print("警告: 未找到漏洞标识列，尝试使用第一个数值列")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                vul_col = numeric_cols[0]
            else:
                raise ValueError("无法找到合适的漏洞标识列")
        
        if code_col is None:
            print("警告: 未找到代码列，尝试使用最长的文本列")
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                # 选择平均长度最长的文本列作为代码列
                max_avg_len = 0
                for col in text_cols:
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > max_avg_len:
                        max_avg_len = avg_len
                        code_col = col
            else:
                raise ValueError("无法找到合适的代码列")
        
        print(f"使用漏洞标识列: {vul_col}")
        print(f"使用代码列: {code_col}")
        
        # 创建新的数据结构
        converted_data = []
        
        for index, row in df.iterrows():
            # 确保漏洞标识为0或1
            target_value = row[vul_col]
            if pd.isna(target_value):
                target_value = 0
            else:
                target_value = int(target_value)
            
            # 获取代码内容
            func_value = row[code_col]
            if pd.isna(func_value):
                func_value = ""
            else:
                func_value = str(func_value)
            
            converted_item = {
                'target': target_value,
                'func': func_value
            }
            converted_data.append(converted_item)
        
        # 统计正负样本
        positive_count = sum(1 for item in converted_data if item['target'] == 1)
        negative_count = len(converted_data) - positive_count
        
        print(f"\n转换后统计:")
        print(f"总样本数: {len(converted_data)}")
        print(f"正样本(有漏洞): {positive_count}")
        print(f"负样本(无漏洞): {negative_count}")
        print(f"正样本比例: {positive_count/len(converted_data):.3f}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        # 保存为JSON Lines文件
        print(f"\n正在保存JSON Lines文件: {json_file_path}")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            for item in converted_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"转换完成！输出文件: {json_file_path}")
        
        return converted_data
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        raise


if __name__ == '__main__':
    # 输入和输出文件路径
    csv_file_path = '../dataset/data/MSR_data_cleaned.csv'
    json_file_path = '../dataset/data/MSR_data_cleaned.json'
    
    # 执行转换
    convert_csv_to_json(csv_file_path, json_file_path)