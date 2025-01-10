%%���쵥�νӴ�����Ȩ����Ӵ���¼
clear all;
close all;
clc;
name='tb';
data=importdata('tbdata.mat');
filename=[name,'PairwiseInteraction.txt'];
fid=fopen(filename,'w');
aplist=unique(data(:,4));%AP������б�
N=size(aplist,1);%AP������б��ģ
    for jj=1:N
        ap=aplist(jj,1);%����һ��AP�����
        index=find(data(:,4)==ap);
        datatmpLength=size(index,1);
        if(datatmpLength<=1)%��ֻ��һ����¼�����޼�¼��Ȳ��ᷢ���Ӵ�
            continue;
        end
        datatmp=data(index,1:3);%��¼��Ϣ����ͬAP��
        datatmp=sortrows(datatmp,3);%�Խ�������ʱ���С��������
        for record=1:datatmpLength-1
            dataleft=datatmp(record+1:datatmpLength,:);
            index=find(dataleft(:,2)<datatmp(record,3) & dataleft(:,3)>=datatmp(record,3));%��������ʱ������һ���ڵ�����ʱ��ķ�Χ�ڣ��������ڵ�ػᷢ���Ӵ�
            if(size(index,1)==0)
                continue;
            end
            temp=dataleft(index,:);%���з��Ϸ����Ӵ������ļ�¼
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