clear all;
close all;
clc;

data=importdata('tbPairwiseInteraction.mat');
dat=data(data(:,3)>210,1:3);
[Ndata,Ddata]=weightGenerationFunction(dat);