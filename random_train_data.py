import os,shutil,random

data_path_0 = './data/data/0/'
data_path_1 = './data/data/1/'
label_path_0 = './data/label/0/'
label_path_1 = './data/label/1/'

train_0 = 790
train_1 = 790
test_0 = 512
test_1 = 88

data_0_list = os.listdir(data_path_0)
random.shuffle(data_0_list)
data_1_list = os.listdir(data_path_1)
random.shuffle(data_1_list)

for i in range(0,train_0):
    shutil.copy(data_path_0+data_0_list[i],"./train_data/data/"+data_0_list[i])
    shutil.copy(label_path_0+data_0_list[i],"./train_data/label/"+data_0_list[i])

for i in range(train_0,train_0 + test_0):
    shutil.copy(data_path_0+data_0_list[i],"./test_data/data/"+data_0_list[i])
    shutil.copy(label_path_0+data_0_list[i],"./test_data/label/"+data_0_list[i])

for i in range(0,train_1):
    shutil.copy(data_path_1+data_1_list[i],"./train_data/data/"+data_1_list[i])
    shutil.copy(label_path_1+data_1_list[i],"./train_data/label/"+data_1_list[i])

for i in range(train_1,train_1+test_1):
    shutil.copy(data_path_1+data_1_list[i],"./test_data/data/"+data_1_list[i])
    shutil.copy(label_path_1+data_1_list[i],"./test_data/label/"+data_1_list[i])