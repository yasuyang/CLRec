
#requirements
pip install fire
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python =3.8
pip install transformers=4.18.0
pip install datasets
pip install scikit-learn
pip install sentencepiece

结构介绍：
训练对象：t5-small
训练数据集：来自亚马逊商品数据集，对比学习数据集 + podmaster论文原始数据集
checkpoint：存放的是不同数据集下需要训练的模型参数，该文件夹下的数据集都是经过对比学习数据集训练过
checkpoint-trained:存放的是checkpoint下的模型经过原始数据集训练过的模型参数
result:存放的是训练俩遍后对seq、exp、topn任务推理后的结果
