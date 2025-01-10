clear all;
close all;
clc;


%%open data set of fudanwifi09

%%access logs: userid, startTime, duration, location


data=importdata('sampledInteractionForMotifs.txt');

fid=fopen('File4_sampledInteractionData.txt','wt');

data(:,3)=data(:,3)./60;

fprintf(fid,'%s\t%s\t%s\n','userId','userId','startTime');
N=size(data,1);
for i=1:N
    fprintf(fid,'%d\t%d\t%d\n',data(i,1),data(i,2),data(i,3));
    N-i
end
fclose(fid);