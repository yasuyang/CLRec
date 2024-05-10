import json
def split_json_into_files(input_filename):
    # 打开并读取输入的JSON文件
    with open(input_filename, 'r') as file:
        data = json.load(file)

    # 检查数据是否为字典且含有三个键值对
    if not isinstance(data, dict) or len(data) != 3:
        raise ValueError("输入的JSON文件必须包含恰好三个键值对")

    # 遍历字典中的键值对，并为每个键值对创建一个新的JSON文件
    train_data = dict()
    valid_data = dict()
    train_data['data'] = data['train']
    valid_data['data'] = data['val']
    with open('../data/toys/exp_com_data/train_data.json', 'w') as file:
        # 将字典数据转换为JSON格式并写入文件
        json.dump(train_data, file, indent=4)  # indent参数使JSON文件格式化，更易于阅读
    with open('../data/toys/exp_com_data/valid_data.json', 'w') as file:
            # 将字典数据转换为JSON格式并写入文件
        json.dump(valid_data, file, indent=4)  # indent参数使JSON文件格式化，更易于阅读
split_json_into_files('../data/toys/exp_com_data/data.json')