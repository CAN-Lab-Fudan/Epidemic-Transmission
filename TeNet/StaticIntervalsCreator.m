clear all;
close all;
clc;

data=importdata('tbPairwiseInteraction.mat');
temp=data(:,2);
temp(:,2)=data(:,1);
temp(:,3)=data(:,4);
data(:,3)=[];
data(:,4)=[];
data=[data;temp];

user=unique(data(:,1));
N=size(user,1);
fid=fopen('staticInterval.txt','w');
for i=1:N
    temp=data(data(:,1)==user(i,1),:);
    temp=sortrows(temp,3);
    M=size(temp,1);
    for j=1:M-1
        if((temp(j,2)==temp(j+1,2))||(temp(j+1,3)-temp(j,3)==0))
            continue;
        end
        fprintf(fid,'%s,%s,%s,%s,%s\n',num2str(temp(j,1)),num2str(temp(j,2)),num2str(temp(j+1,2)),num2str(temp(j,3)),num2str(temp(j+1,3)));
    end
    i
end
fclose(fid);