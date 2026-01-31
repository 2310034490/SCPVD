import json
import random
import os


def load_devign_dataset(file_path):
    """加载devign数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def split_dataset(data, train_ratio=0.8, val_ratio=0.1):
    """将数据集分割为训练集、验证集和测试集，并保持正负样本比例"""
    positive_samples = [item for item in data if item.get('target') == 1]
    negative_samples = [item for item in data if item.get('target') == 0]

    # 计算每个类别应有的训练、验证、测试集大小
    n_pos = len(positive_samples)
    n_neg = len(negative_samples)

    train_pos_size = int(n_pos * train_ratio)
    val_pos_size = int(n_pos * val_ratio)
    test_pos_size = n_pos - train_pos_size - val_pos_size

    train_neg_size = int(n_neg * train_ratio)
    val_neg_size = int(n_neg * val_ratio)
    test_neg_size = n_neg - train_neg_size - val_neg_size

    # 分割正样本
    train_pos = positive_samples[:train_pos_size]
    val_pos = positive_samples[train_pos_size:train_pos_size + val_pos_size]
    test_pos = positive_samples[train_pos_size + val_pos_size:train_pos_size + val_pos_size + test_pos_size]

    # 分割负样本
    train_neg = negative_samples[:train_neg_size]
    val_neg = negative_samples[train_neg_size:train_neg_size + val_neg_size]
    test_neg = negative_samples[train_neg_size + val_neg_size:train_neg_size + val_neg_size + test_neg_size]

    # 合并并打乱（可选，但为了保持比例，通常不打乱）
    train_data = train_pos + train_neg
    val_data = val_pos + val_neg
    test_data = test_pos + test_neg

    # 再次打乱以混合正负样本，但保持内部比例
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def process_data_items(data_list, start_idx=0):
    """处理数据项，删除不需要的字段并添加idx"""
    processed_data = []
    for i, item in enumerate(data_list):
        # 只保留func和target字段，添加idx
        processed_item = {
            'idx': start_idx + i,
            'func': item.get('func', ''),
            'target': item.get('target', 0)
        }
        processed_data.append(processed_item)
    return processed_data


def save_jsonl(data, output_file):
    """将数据保存为jsonl格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def process_devign_dataset(input_file, output_dir):
    """处理devign数据集并保存为训练、验证和测试集"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    data = load_devign_dataset(input_file)

    # 分割数据集
    train_data, val_data, test_data = split_dataset(data)

    # 处理数据项，删除不需要的字段并添加连续的idx
    processed_train_data = process_data_items(train_data, start_idx=0)
    processed_val_data = process_data_items(val_data, start_idx=len(processed_train_data))
    processed_test_data = process_data_items(test_data, start_idx=len(processed_train_data) + len(processed_val_data))

    # 保存分割后的数据集
    save_jsonl(processed_train_data, os.path.join(output_dir, 'train_devign.jsonl'))
    save_jsonl(processed_val_data, os.path.join(output_dir, 'valid_devign.jsonl'))
    save_jsonl(processed_test_data, os.path.join(output_dir, 'test_devign.jsonl'))

    # 打印数据集统计信息
    print(f"数据集总大小: {len(data)}")
    print(f"训练集大小: {len(processed_train_data)}")
    print(f"验证集大小: {len(processed_val_data)}")
    print(f"测试集大小: {len(processed_test_data)}")
    print(f"训练集idx范围: 0-{len(processed_train_data) - 1}")
    print(f"验证集idx范围: {len(processed_train_data)}-{len(processed_train_data) + len(processed_val_data) - 1}")
    print(
        f"测试集idx范围: {len(processed_train_data) + len(processed_val_data)}-{len(processed_train_data) + len(processed_val_data) + len(processed_test_data) - 1}")


if __name__ == '__main__':
    input_file = '../dataset/data/devign.json'
    output_dir = '../dataset/data/devign'
    process_devign_dataset(input_file, output_dir)