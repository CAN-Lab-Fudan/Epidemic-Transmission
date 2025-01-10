clear all;
close all;
clc;


baseline=1255795200;%%datevec(datenum([1970 1 1 0 0 baseline]))=16:00/17/10/2009+ 8H =18/10/2009
data=importdata('records.mat');
tbaps=importdata('tbaps.txt');
M=size(tbaps,1);
fid=fopen('tbdata.txt','w');
for i=1:M
    temp=data(data(:,4)==tbaps(i,1),1:3);
    temp(:,4)=tbaps(i,2);
    for d=1:84
        begin=baseline+24*3600*(d-1);
        finish=baseline+24*3600*d;
        tmp=temp(temp(:,2)>begin & temp(:,3)<finish,:);
        tmp(:,2)=tmp(:,2)-baseline;
        tmp(:,3)=tmp(:,3)-baseline;
        N=size(tmp,1);
        for j=1:N
            fprintf(fid,'%s,%s,%s,%ld\n',num2str(tmp(j,1)),num2str(tmp(j,2)),num2str(tmp(j,3)),tmp(j,4));
        end
    end    
    i
end
fclose(fid);

tbdata=importdata('tbdata.txt');
tbdata=sortrows(tbdata,2);

[user,index]=unique(tbdata,'first');
user(:,2)=index;
user=sortrows(user,2);
N=size(user,1);
for j=1:N%给用户标号，按用户在数据中出现的先后次序
    index=find(tbdata(:,1)==user(j,1));
    tbdata(index,5)=j;
    j
end
tbdata(:,1)=tbdata(:,5);
tbdata(:,5)=[];
save('tbdata.mat',tbdata);


