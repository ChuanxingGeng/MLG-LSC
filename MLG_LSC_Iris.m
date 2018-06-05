%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A demo of MLG-LSC for multiclass classification written by Chuanxing Geng
% For any problem concerning the codes, please feel free to contact me (gengchuanxing@126.com or 
% gengchuanxing@nuaa.edu.cn).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc;
load('Iris.mat');
Ylabel = Iris(:,end-2:end);
lamda = 0.01;
mu = 0.001;
alpha = 0.001;
time = zeros(1,10);
Accuracy = zeros(1,10);
for j = 1:10;   % Randomly repeat 10 rounds
% Data preprocessing
% Selecting the class 1 dataset
indx1 = find(Ylabel(:,1)==1);
X1data = Iris(indx1,:);
K1 = randperm(size(X1data,1));
num1 = round(size(X1data,1)*0.4);
% Selecting the class 2 dataset
indx2 = find(Ylabel(:,2)==1);
X2data = Iris(indx2,:);
K2 = randperm(size(X2data,1));
num2 = round(size(X2data,1)*0.4);
% Selecting the class 3 dataset
indx3 = find(Ylabel(:,3)==1);
X3data = Iris(indx3,:);
K3 = randperm(size(X3data,1));
num3 = round(size(X3data,1)*0.4);
%%% The number of total training samples
Num = num1+num2+num3;

TrainData = zscore([X1data(K1(1:num1),1:end-3);X2data(K2(1:num2),1:end-3);X3data(K3(1:num3),1:end-3)]);
TestData = zscore([X1data(K1(num1+1:end),1:end-3);X2data(K2(num2+1:end),1:end-3);X3data(K3(num3+1:end),1:end-3)]);
Y_train = [X1data(K1(1:num1),end-2:end);X2data(K2(1:num2),end-2:end);X3data(K3(1:num3),end-2:end)];
Y_test = [X1data(K1(num1+1:end),end-2:end);X2data(K2(num2+1:end),end-2:end);X3data(K3(num3+1:end),end-2:end)];
[n,m] = size(TrainData);

%%%----------------------------------Training-----------------------------------------------------
tic;
% Solving W and t in LSR
H = eye(n) - (1/n)*ones(n,1)*ones(n,1)';
W = inv(TrainData'*H*TrainData + lamda*eye(m))*TrainData'*H*Y_train;
t = (1/n)*(Y_train'*ones(n,1)-W'*TrainData'*ones(n,1));

% Solving the LSR error for each class.
C1 = TrainData(1:num1,:)*W + eye(num1,1)*t' - Y_train(1:num1,:);
D1 = TrainData(num1+1:end,:)*W + eye(Num-num1,1)*t' - Y_train(num1+1:end,:);
G1 = C1'*C1 ;
H1 = D1'*D1 ;

C2 = TrainData(num1+1:num1+num2,:)*W + eye(num2,1)*t' - Y_train(num1+1:num1+num2,:);
D2 = [TrainData(1:num1,:);TrainData(num1+num2+1:end,:)]*W + eye(Num-num2,1)*t' - [Y_train(1:num1,:);Y_train(num1+num2+1:end,:)];
G2 = C2'*C2 ;
H2 = D2'*D2 ;

C3 = TrainData(num1+num2+1:end,:)*W + eye(num3,1)*t' - Y_train(num1+num2+1:end,:);
D3 = TrainData(1:num1+num2,:)*W + eye(Num-num3,1)*t' - Y_train(1:num1+num2,:);
G3 = C3'*C3 ;
H3 = D3'*D3 ;

Gz = G1+G2+G3 + mu*eye(size(C1,2));
Hz = H1+H2+H3 + mu*eye(size(C1,2));

% Solving the dragging matrix
A_alpha = CS_GeometricMean(inv(Gz),Hz,alpha);
time(j) = toc;

%%%-----------------------------Prediction---------------------------------------------------------
Y = eye(3);
error = 0;
for i = 1:size(TestData,1)
    dist1 = (TestData(i,:)*W + t' - Y(1,:))*A_alpha*(TestData(i,:)*W + t' - Y(1,:))';
    dist2 = (TestData(i,:)*W + t' - Y(2,:))*A_alpha*(TestData(i,:)*W + t' - Y(2,:))';
    dist3 = (TestData(i,:)*W + t' - Y(3,:))*A_alpha*(TestData(i,:)*W + t' - Y(3,:))';
    L = [dist1, dist2, dist3];
    indx = find(L == min(L));
    predict_label = Y(indx,:); 
    if (norm(predict_label - Y_test(i,:))~= 0)
        error = error +1;
    end
end
Accuracy(j) = 1-error/size(TestData,1);
end
Accuracy_mean = mean(Accuracy)
Std = std(Accuracy)
Elapsedtime = mean(time)

