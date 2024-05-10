
#{"task": "mindsequential", "data_id": 0, "instruction":
# "Considering {dataset} user_{user_id} has interacted with {dataset} items {history} . What is the next recommendation for the user ?",
# "input": "Considering mind user_1 has interacted with mind items item_1001 . What is the next recommendation for the user ?",
# "output": "mind item_1002"}
###序列推荐
seq_templates_2 = '''
Which of the following sequences is more likely to be an interaction sequence for {}?
  1.{}
2.{}
'''

seq_templates_3 = '''
Which of the following sequences is more likely to be an interaction sequence for {}?
1.{}
2.{}
3.{}
'''

#推荐解释
exp_templates_2 = '''
Which of the following ratings is more likely to be made by {} on {}? 1.{}  2.{}
'''
exp_templates_3 = '''
Which of the following ratings is more likely to be made by {} on {}?    
1.{}
2.{}
3.{}
'''

#top-n解释
topn_templates_2 = '''
Which item_{}, the user may hate more?
1.{}
2.{}
'''