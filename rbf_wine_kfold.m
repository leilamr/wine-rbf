%% MLP - dataset: Wine
%author = @leilamr
close all; 
clear all;
clc

%load dataset
[input, target] = wine_dataset;
input = input';
target = target';
inputData = [input,target];

%% cross-validation: KFold = 10 (manual)
j = 10;
avgTest = 0;
% k1:9 = (18x16), k10 = (16x16)
k1 = [inputData(1:10,:); inputData(101:108,:)];
k2 = [inputData(11:20,:); inputData(109:116,:)];
k3 = [inputData(21:30,:); inputData(117:124,:)];
k4 = [inputData(31:40,:); inputData(125:132,:)];    
k5 = [inputData(41:50,:); inputData(133:140,:)];
k6 = [inputData(51:60,:); inputData(141:148,:)];
k7 = [inputData(61:70,:); inputData(149:156,:)];
k8 = [inputData(71:80,:); inputData(157:164,:)];
k9 = [inputData(81:90,:); inputData(165:172,:)];
k10 = [inputData(91:100,:); inputData(173:178,:)];

%above folds exchange
for i=1:j
    if i == 1
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k7;k8;k9];
        testMatrix = k10;
    end
    
    if i == 2
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k7;k8;k10];
        testMatrix = k9;
    end
    
    if i == 3
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k7;k9;k10];
        testMatrix = k8;
    end
    
    if i == 4
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k3;k4;k5;k6;k8;k9;k10];
        testMatrix = k7;
    end
    
     if i == 5
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k3;k4;k5;k7;k8;k9;k10];
        testMatrix = k6;
     end
    
      if i == 6
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k3;k4;k6;k7;k8;k9;k10];
        testMatrix = k5;
      end
      
       if i == 7
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k3;k5;k6;k7;k8;k9;k10];
        testMatrix = k4;
       end
    
     if i == 8
        %make train matrix(160x16)
        trainMatrix = [k1;k2;k4;k5;k6;k7;k8;k9;k10];
        testMatrix = k3;
     end
    
      if i == 9
        %make train matrix(160x16)
        trainMatrix = [k1;k3;k4;k5;k6;k7;k8;k9;k10];
        testMatrix = k2;
      end
    
       if i == 10
        %make train matrix(160x16)
        trainMatrix = [k2;k3;k4;k5;k6;k7;k8;k9;k10];
        testMatrix = k1;
       end
    
       %% start MLP
       %matrix for Train (160x13)
       inputTrain = trainMatrix(:,1:13);
       
       %target (18x3)
       target = trainMatrix(:,14:16);
       
       %matrix transposed input and output network
       inputTrain = inputTrain'; %(4x135)
       inputTrain = mapminmax(inputTrain);
       
       outputTrain = target'; %(3x135)
       
       %% make network
       % Mean squared error goal (default = 0.0)
       goal = 0;
       
       %Spread of radial basis functions (default = 1.0)
       spread = 2.550;

        %Maximum number of neurons
        MN = 10;                                     

        %Number of neurons to add between displays           
        DF = 1;                                 
    
       %net = feedforwardnet(MN);
                                
       %% train network
       net = newrb(inputTrain,outputTrain,goal,spread,MN,DF);
      
       %% test network performance
       inputTest = testMatrix(:,1:13); %(15x4)
       targetTest = testMatrix(:,14:16);
       
       %matrix transposed input and output network
       inputTest = inputTest'; %(4x15)
       inputTest = mapminmax(inputTest);
       
       outputTest = targetTest'; %(3x15)
       
       %network prediction result
       result = sim(net,inputTest);
       [per con] = confusion(outputTest,result);
       perTest = 100 * (1 - per);
       
       avgTest = avgTest + perTest;
       
       %plot and save confusion matrix
       
        x(i)=plotconfusion(result,outputTest);
    
        if i==1
            saveas(x(i),'confusion_1.jpg');  
        end
        
        if i==2
            saveas(x(i),'confusion_2.jpg');  
        end
        
        if i==3
            saveas(x(i),'confusion_3.jpg');  
        end
        
        if i==4
            saveas(x(i),'confusion_4.jpg');  
        end
        
        if i==5
            saveas(x(i),'confusion_5.jpg');  
        end
        
        if i==6
            saveas(x(i),'confusion_6.jpg');  
        end
        
        if i==7
            saveas(x(i),'confusion_7.jpg');  
        end
        
        if i==8
            saveas(x(i),'confusion_8.jpg');  
        end
        
        if i==9
        saveas(x(i),'confusion_9.jpg');  
        end
        
        if i==10
            saveas(x(i),'confusion_10.jpg');  
        end
end
 avgTest=avgTest/10;
    
 fprintf('network hit average (kfold = 10): %.3f%%\n', avgTest);
