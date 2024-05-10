#requirments.txt
pip install fire
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python =3.8
pip install transformers=4.18.0
pip install datasets
pip install scikit-learn
pip install sentencepiece

结构介绍：
本项目是利用构造的对比学习数据集对t5进行训练，以方便下游任务用原始数据集进行微调，增强其能力

checkpoint存放的是各个数据集通过对比训练后的参数