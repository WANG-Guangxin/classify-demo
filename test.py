import torch
import os
import scipy.io as io
import model
import time

def test(test_data_path, test_label_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_list = os.listdir(test_data_path)

    print("测试集数据：",len(data_list))

    f = open('./test_result.csv','w')
    result = 'data_name,label,result\n'

    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for name in data_list:
        result += name
        result += ','
        data = io.loadmat(test_data_path+name)
        label = io.loadmat(test_label_path+name)
        data = data['data'] / 1.0
        label = label['label'] / 1.0
        result += str(label[0][0])
        result += ','
        data_tensor = torch.tensor(data).float().unsqueeze(0).cuda().unsqueeze(0)
        label_tensor = torch.tensor(label).float().unsqueeze(0).cuda().unsqueeze(0)
        net = model.MyNet().cuda()
        net.load_state_dict(torch.load('final_weight/final.pth'))
        print(name)

        out = net(data_tensor)
        print(out)

        if(out[0][0] < 0.5):
            out[0][0] = 0.0
        else:
            out[0][0] = 1.0

        result += str(int(out[0][0]))
        result += '\n'

        if(label[0][0] == 1.0 and out[0][0] == 1.0):
            TP += 1
        if (label[0][0] == 1.0 and out[0][0] == 0.0):
            FN += 1
        if (label[0][0] == 0.0 and out[0][0] == 1.0):
            FP += 1
        if (label[0][0] == 0.0 and out[0][0] == 0.0):
            TN += 1

    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)
    jql = (TP) / (TP + FP)
    zhl = (TP) / (TP + FN)
    print("准确率：", (TP + TN)/(TP + FN +TN +FP))
    print("精确率：", jql)
    print("召回率：", zhl)
    print("F1 score：", (2 * jql * zhl)/(jql+zhl) )
    result += 'TP,'
    result += (str(TP)+'\n')
    result += 'FP,'
    result += (str(FP) + '\n')
    result += 'TN,'
    result += (str(TN) + '\n')
    result += 'FN,'
    result += (str(FN) + '\n')
    result += '准确率,'
    result += (str((TP + TN)/(TP + FN +TN +FP)) + '\n')
    result += '精确率,'
    result += (str(jql) + '\n')
    result += '召回率,'
    result += (str(zhl) + '\n')
    result += 'F1 score,'
    result += (str((2 * jql * zhl)/(jql+zhl)) + '\n')

    f.write(result)
    f.close()






if __name__ == '__main__':
    with torch.no_grad():
        test_data_path = './test_data/data/'
        test_label_path = './test_data/label/'

        test(test_data_path,test_label_path)