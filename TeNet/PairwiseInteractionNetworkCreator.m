%%构造单次接触的有权无向接触记录
clear all;
close all;
clc;
name='tb';
data=importdata('tbdata.mat');
filename=[name,'PairwiseInteraction.txt'];
fid=fopen(filename,'w');
aplist=unique(data(:,4));%AP接入点列表
N=size(aplist,1);%AP接入点列表规模
    for jj=1:N
        ap=aplist(jj,1);%其中一个AP接入点
        index=find(data(:,4)==ap);
        datatmpLength=size(index,1);
        if(datatmpLength<=1)%若只有一条记录或则无记录则比不会发生接触
            continue;
        end
        datatmp=data(index,1:3);%记录信息（相同AP）
        datatmp=sortrows(datatmp,3);%以结束上网时间从小到达排列
        for record=1:datatmpLength-1
            dataleft=datatmp(record+1:datatmpLength,:);
            index=find(dataleft(:,2)<datatmp(record,3) & dataleft(:,3)>=datatmp(record,3));%结束上网时间在另一个节点上网时间的范围内，则两个节点必会发生接触
            if(size(index,1)==0)
                continue;
            end
            temp=dataleft(index,:);%所有符合发生接触条件的记录
            tempLength=size(temp,1);
            for id=1:tempLength
                tempuser=unique([datatmp(record,1),temp(id,1)]);
                if(size(tempuser,2)==2)
                    accesstime=max(datatmp(record,2),temp(id,2));
                    breaktime=min(datatmp(record,3),temp(id,3));
                    duration=(breaktime-accesstime)/60;
                    fprintf(fid,'%s %s %s %s %s\n',num2str(tempuser(1,1)),num2str(tempuser(1,2)),num2str(duration),num2str(accesstime),num2str(ap));
               end
            end
        end
        jj
    end
fclose(fid);

interactionTime=importdata(filename);
interaction=interactionTime(:,1:3);
[contactNumber,contactDuration]=weightGenerationFunction(interaction);
save ([name,'PairwiseInteraction.mat'], 'interactionTime');
save ([name,'AggregatedDuration.mat'], 'contactDuration');
save ([name,'AggregatedNumber.mat'], 'contactNumber');