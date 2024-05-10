import os
import random
import sys
sys.path.append('../')
import templates
import json

def read_line(path):
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines
def construct_user_sequence_dict(user_sequence):
    """
    Convert a list of string to a user sequence dict. user as key, item list as value.
    """
    user_seq_dict = dict()
    for line in user_sequence:
        user_seq = line.split(" ")
        user_seq_dict[user_seq[0]] = user_seq[1:]
    return user_seq_dict
def load_prompt_template(prompt_file, task, con_num):

    template_name = f'{task}_templates_{con_num}'
    try:
        template = getattr(templates,template_name)
    except AttributeError as e:
        raise ValueError(f'Template {template_name} not found in templates module.') from e
    return template

def choice_con(training_data_samples,batch_size,current_id):
    selected_elements = []
    while len(selected_elements) < (batch_size-1):
        # 随机选择一个元素
        random_dict = random.choice(training_data_samples)
        # 如果选择的元素包含特定值，跳过该元素
        if random_dict['user_id'] == current_id:
            continue
        # 将不包含特定值的元素添加到结果列表中
        selected_elements.append(random_dict)
    return selected_elements


def split_json_file(input_file_path, output_file_path1, output_file_path2, data_point,ratio):
    # 读取原始的JSON文件
    random.shuffle(data_point)

    # 计算分割点
    split_point = int(0.7 * len(data_point))

    # 分割数据
    data_part1 = data_point[:split_point]
    data_part2 = data_point[split_point:]
    train = dict()
    train['data'] = data_part1
    val = dict()
    val['data'] = data_part2
    # 保存第一部分数据到JSON文件
    with open(output_file_path1, 'w') as f:
        json.dump(train, f)

    # 保存第二部分数据到JSON文件
    with open(output_file_path2, 'w') as f:
        json.dump(val, f)

def get_exp_data(data_path):
    modes=['train','val','test']
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result = dict()
    for mode in modes:
        # 获取字典列表
        temp = data[mode]
        temp_list=[]
        while len(temp) > 3:
            current_dict = temp.pop(0)
            current_dict_com = get_com_dict(temp,current_dict)
            if current_dict_com ==None:
                continue
            data_one, data_two = process_pair(current_dict,current_dict_com)
            temp_list.append(data_one)
            temp_list.append(data_two)
            temp.remove(current_dict_com)
        result[mode] = temp_list
    return result


def get_com_dict(current_list, current_dict):
    feature_value = current_dict.get('feature')
    reference_explanation_length = len(current_dict.get('explanation'))

    # 过滤出具有相同 'feature' 值的所有字典
    matching_dicts = [d for d in current_list if d.get('feature') == feature_value]

    if len(matching_dicts) ==0 or(len(matching_dicts)==1 and matching_dicts[0].get('explanation')==current_dict['explanation']):
        closest_dict = min(
            current_list,
            key=lambda x: abs(len(x.get('explanation')) - reference_explanation_length)
        )
        if closest_dict == None:
            print(current_list)
            print('-------------------')
            print(current_dict)
        return closest_dict

    # 根据 'explanation' 长度进行排序
    sorted_dicts = sorted(
        matching_dicts,
        key=lambda x: abs(len(x.get('explanation')) - reference_explanation_length)
    )

    # 找到第一个与参考值不同的字典
    for dict_ in sorted_dicts:
        if dict_.get('explanation') != current_dict['explanation']:
            return dict_


    # 返回最接近长度的 'explanation' 值
def process_pair(current_dic,com_dic):
    data1 =dict()
    template = templates.exp_templates_2
    # print(current_dic)
    # print(com_dic)
    formatted_string = template.format('user_'+str(current_dic['user']), 'item_'+str(current_dic['item']), current_dic['explanation'], com_dic['explanation'])
    data1['input'] = formatted_string
    data1['output'] = current_dic['explanation']

    data2 =dict()
    formatted_string1 = template.format('user_'+str(com_dic['user']), 'item_'+str(com_dic['item']), com_dic['explanation'], current_dic['explanation'])
    data2['input'] = formatted_string1
    data2['output'] = com_dic['explanation']

    return data1,data2