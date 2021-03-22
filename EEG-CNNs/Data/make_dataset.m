clear all
clc

format long

training_set = [];
training_label   = [];
num_subject   = 15;
num_trial     = 150;
num_channel   = 60;
num_data      = 2501;
%Time_consider = 4 / 10;
%Data_points   = Time_consider * 160;

%% Read the Data and Create Train Labels
for i = 1:15
    Dataset = ['sample', num2str(i), '.mat'];
    Dataset = load(Dataset);
    Label = Dataset.epo.y;
    Label = Label.';
    training_label = [training_label;Label];
end

%% Read and Create Train Data
for i = 1:15
    Dataset = ['sample', num2str(i), '.mat'];
    Dataset = load(Dataset);
    Data = Dataset.epo.x(1502:2501,:,:);
    
    Data_zero = [];
    
    for j =1:150
        Datas = Data(:,:,j); %1000*60       
        Datas = reshape(Datas,60000,1);
        Datas = Datas.';
        Data_zero = [Data_zero;Datas];
    end
    
    training_set = [training_set;Data_zero];
end

%%
[m,n] = size(training_set);
for i=1:n
    mean_x = mean(training_set(:,i));
    std_x = std(training_set(:,i));
    training_set(:,i) = (training_set(:,i)-mean_x)/std_x;
end
%%
Dataset = [training_set, training_label];
[m, n] = find(isnan(Dataset));
Dataset(m, :) = [];
rowrank = randperm(size(Dataset, 1)); 
Dataset = Dataset(rowrank, :); 

%%
Training_set_15 = Dataset(:, 60001:60003);
csvwrite('training_label_15.csv', Training_set_15);

%%
Training_data_15 = Dataset(:, 1:6000);
%%
csvwrite('training_set_15.csv', Training_data_15);

