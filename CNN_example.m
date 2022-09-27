 
% This pseudocode is designed to implement the CNN for signal processing by Nadia Aghili

%% Train CNN
Xtr_1 = feature_train; % feature_train size = [channel, temporal, trials]
XTrain=[];
for i=1:size(Xtr_1,3)
    m=Xtr_1(:,:,i);
    XTrain(1:size(Xtr_1,1),1:size(Xtr_1,2),1,i)=m;
end
Label = categorical(Y); % Y is the classes labels that the size is based on trials

layers = [ ...
        imageInputLayer([size(XTrain,1),size(XTrain,2),1])
        % you can use different number, size, and stride based on your
        % input size in your convulotion layers
        convolution2dLayer([size(XTrain,1)   1],20,'Stride',1) 
        convolution2dLayer([1 6],20,'Stride',1)
        batchNormalizationLayer
        reluLayer
        dropoutLayer 
        fullyConnectedLayer(2) % you can used several fully connected but in the last fully connected befor softmax, you must set the number of neuron based on number of classes  
        softmaxLayer
        classificationLayer];
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.001, ...
        'Verbose',false, ...
        'Shuffle','every-epoch');
    net = trainNetwork(XTrain,Label,layers,options);
    K = 1; % it is the CNN layers
    features_CNN = activations(net,XTrain,K); % this line shows the CNN layers feature map
    sz = size(features_CNN) % this line shows the size of the features_CNN
    
 %% Test CNN
Xtr_2 = feature_test; % feature_train size = [channel, temporal, trials]
XTest=[];
for i=1:size(Xtr_2,3)
    m=Xtr_2(:,:,i);
    XTest(1:size(Xtr_2,1),1:size(Xtr_2,2),1,i)=m;
end
[YPred,scores] = classify(net,XTest);
 
    
    