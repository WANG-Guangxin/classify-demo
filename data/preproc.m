clc;clear;close all
load("data_TR.mat")

I_2020 = imread('2020DL.tif');
I_2022 = imread('2022DL.tif');

a = 0;b = 0;
data = zeros([2,32]);
label = [];
for i = 1:150
    for j = 1:400
        if(data_TR(i,j) ~= 0)
            if(data_TR(i,j) == 1.0)
                label = 0.0;
                save(['./label/0/',num2str(i),'_',num2str(j),'.mat'],"label")
                a = a + 1;
            end
            if(data_TR(i,j) == 2.0)
                label = 1.0;
                save(['./label/1/',num2str(i),'_',num2str(j),'.mat'],"label")
                b = b + 1;
            end
            for k = 1:32
                data(1,k) = I_2020(i,j,k);
                data(2,k) = I_2022(i,j,k);
            end
            if(label == 0.0)
                save(['./data/0/',num2str(i),'_',num2str(j),'.mat'],"data")
            end
            if(label == 1.0)
                save(['./data/1/',num2str(i),'_',num2str(j),'.mat'],"data")
            end
        end
    end
end

