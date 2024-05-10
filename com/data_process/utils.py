import os
import random
import sys
import templates
import json
import csv
sys.path.append('../')

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

def Csv2Json(csv_file,map_file = None):
    data = {}  # 创建一个空的字典用于存储数据
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)  # 使用reader读取CSV文件
        for row in reader:
            user_id = row[0]  # 假设用户ID在第一列
            item_id = row[1]  # 假设项目ID在第二列
            rating = float(row[2])  # 假设评分在第三列，并将其转换成浮点数
            # 如果用户ID不存在于字典中，则创建一个新的键值对
            if user_id not in data:
                data[user_id] = []
            # 添加评分信息到对应用户的列表中
            data[user_id].append({'itemID': item_id, 'rating': rating})
    #转换id对
    if map_file is not None:
        id_map_data = {}
        with open(map_file, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        id2user = {v: k for k, v in id_mapping["id2user"].items()}
        id2item = {v: k for k, v in id_mapping["id2item"].items()}
        #开始匹配
        for old_key, value in data.items():
            if old_key in id2user:
                new_key = id2user[old_key]
            else:
                #new_key = old_key
                continue
            new_value = []
            for item in value:
                if item["itemID"] in id2item:
                    item["itemID"] = id2item[item["itemID"]]
                    new_value.append(item)
                else:
                    continue
            if len(new_value) > 0:
                id_map_data[new_key] = new_value
        #更换指针
        data = id_map_data
    return data

def Slect_TopN(data:dict,rate = 4.0,exceed = True):
    filtered_data = {}
    for user_id, ratings in data.items():
        if exceed:
            filtered_ratings = [rating for rating in ratings if rating['rating'] >= rate]
        else:
            filtered_ratings = [rating for rating in ratings if rating['rating'] < rate]
        # 如果筛选后的评分列表不为空，则将其添加到筛选后的数据中
        if filtered_ratings:
            filtered_data[user_id] = filtered_ratings
    return filtered_data

def Struct_TopN(data_path,rate = 4.0,map_path = None,Save_Path = "./"):
    data = Csv2Json(data_path,map_path)
    positive_data = Slect_TopN(data,rate,True)
    negetive_data = Slect_TopN(data, rate, False)
    #开始构造
    struct_positive_temp = []
    struct_negetive_temp = []
    tempplate = '''Choose the best item from the candidates to recommend for user_{}? {}'''
    #构造负样本
    for user_id, item in positive_data.items():
        part_items = []
        if user_id in negetive_data:
            for i in positive_data[user_id]:
                part_items.append(i["itemID"])
        for user_id_item in item:
            if len(part_items) > 15:
                random.shuffle(part_items)
                temp_items = part_items[0:14].copy()
            else:
                random.shuffle(part_items)
                temp_items = part_items.copy()
            temp_items.append(user_id_item["itemID"])
            if len(temp_items) > 0:
                random.shuffle(temp_items)
            item_sentense = ""
            for index,j in enumerate(temp_items):
                item_sentense += "{}.item_{} ".format(index+1,j)
                if j == user_id_item["itemID"]:
                    output_sentense = "item_{}".format(j)
            input_sentense = tempplate.format(user_id,item_sentense)
            struct_negetive_temp.append({"input":input_sentense,"output":output_sentense})
    #构造正样本
    for user_id, item in positive_data.items():
        part_items = []
        if user_id in positive_data:
            for i in positive_data[user_id]:
                part_items.append(i["itemID"])
        temp_items = part_items.copy()
        for user_id_item in item:
            temp_items.append(user_id_item["itemID"])
            if len(temp_items) > 0:
                random.shuffle(temp_items)
            item_sentense = ""
            for j in temp_items:
                item_sentense += "item_{},".format(j)
            input_sentense = tempplate.format(user_id, item_sentense)
            output_sentense = user_id_item
            struct_positive_temp.append({"input": input_sentense, "output": output_sentense})

    #保存文件
    json_positive_data = {"train":struct_positive_temp}
    json_negetive_data = {"train":struct_negetive_temp}
    posivtive_file = os.path.join(Save_Path,'Positive_TopN.json')
    negetive_file = os.path.join(Save_Path,'Negetive_TopN.json')
    # with open(posivtive_file, 'w', encoding='utf-8') as posivtive_f:
    #     json.dump(json_positive_data, posivtive_f, ensure_ascii=False, indent=4)
    with open(negetive_file, 'w', encoding='utf-8') as f:
        json.dump(json_negetive_data, f, ensure_ascii=False, indent=4)
    return json_positive_data,json_negetive_data


