import os
import json
import sys
import fire
import random
from utils import construct_user_sequence_dict,read_line
from utils import load_prompt_template,choice_con,split_json_file,get_exp_data


def main(data_dir:str, task:str, dataset:str, prompt_file:str, con_num:int, batch_size:int, output_path:str,train_data:str, val_data:str,ratio:float):
    file_data = dict()
    file_data['arguments'] = {
        "data_dir": data_dir,"task":task,"dataset":dataset,"prompt_file":prompt_file
    }
    file_data['data'] = [0]
    if task == 'seq':
        data_path = f'{data_dir}/{dataset}/sequential.txt'
        user_sequence = read_line(data_path)
        user_sequence_dict = construct_user_sequence_dict(user_sequence)

        # get prompt
        prompt = load_prompt_template(prompt_file, task, con_num)
        training_data_samples = []
        for user in user_sequence_dict:
            items = user_sequence_dict[user][:]
            one_sample =dict()
            one_sample['user_id'] = user
            one_sample['target'] = items
            training_data_samples.append(one_sample)

        print("load training data")
        print(f'there are {len(training_data_samples)} samples in training data.')

        #construct sentences
        data_point= []
        for one in training_data_samples:
            dict_list = choice_con(training_data_samples,batch_size,one['user_id'])
            for i in dict_list:
                # 处理列表数据：将列表中的每个元素前加上 'item_' 并使用空格连接成一个字符串
                formatted_target = " ".join(f"item_{item}" for item in one['target'])+'  '
                # 处理字符数据：将字符数据前加上 'user_'

                formatted_user_id = f"user_{one['user_id']}"
                formatted_target_com = " ".join(f"item_{item}" for item in i['target'])
                formatted_template = prompt.format(formatted_user_id, formatted_target,formatted_target_com)
                temp = dict()
                temp['input'] = formatted_template.replace("\n", "")
                temp['output'] = formatted_target
                data_point.append(temp)
        print("data constructed")
        print(f"there are {len(data_point)} prompts in training data.")
        split_json_file(data_path, train_data, val_data,data_point,ratio)
    elif task =='exp':
        data_path = f'{data_dir}/{dataset}/explanation.json'
        data = get_exp_data(data_path)#得到一个字典
        with open(output_path, 'w') as f:
            # 使用 json.dump() 函数将字典写入文件
            json.dump(data, f)

if __name__ == "__main__":
    main('../data','exp','sports','../templates.py',2,8,'../data/sports/exp_com_data/data.json',
         '../data/sports/exp_com_data/train_data.json','../data/sports/exp_com_data/valid_data.json',0.9)




