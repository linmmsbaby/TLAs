function [fileName, testError] = bayesianOPT_of_CNN_or_GRU_or_BiLSTM(XTrain, YTrain, XTest, YTest, CNN_OR_GRU_OR_BiLSTM)
% Arrangement, modification and annotation by Mansheng Lin
% ANNOTATION:
% INPUT variables
% XTrain —— the input training data, it is 4-D format, each row is each feature, each colum is each sample
% YTrain —— the output training data, it is categories
% XTest —— the input testing data, it is 4-D format, each row is each feature, each colum is each sample
% YTest —— the output testing data, it is categories
% CNN_OR_GRU_OR_BiLSTM —— 1,2,3 represent the deep learning framework is cnn, gru and bilstm, respectively   
% OUTPUT variables
% fileName —— the best BayesObject file name
% testError —— the best net framework test error of testing dataset


if CNN_OR_GRU_OR_BiLSTM == 1
    optimVars = [
        optimizableVariable('SectionDepth',[1 3],'Type','integer')
        optimizableVariable('filterSize1',[2 5],'Type','integer')
        optimizableVariable('filterSize2',[2 5],'Type','integer')
        optimizableVariable('filterSize3',[2 5],'Type','integer')
        optimizableVariable('InitialLearnRate',[1e-3 1],'Transform','log')
        optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')
        optimizableVariable('miniBatchSize',[6 40],'Type','integer')
        optimizableVariable('LearnRateDropPeriod',[10 40],'Type','integer')
        optimizableVariable('LearnRateDropFactor',[0.1 0.9],'Type','real')
        optimizableVariable('dropoutrate',[0.1 0.7],'Type','real')
    ];
else
    XTrain = reshape(XTrain,[size(XTrain,1),size(XTrain,4)]);
    XSum = cell(size(XTrain,2),1);
    for i = 1:size(XSum,1)
         XSum{i} =  XTrain(:,i);
    end
    XTrain = XSum;
    XTest = reshape(XTest,[size(XTest,1),size(XTest,4)]);
    XSum = cell(size(XTest,2),1);
    for i = 1:size(XSum,1)
         XSum{i} =  XTest(:,i);
    end
    XTest = XSum;
    
    optimVars = [
        optimizableVariable('SectionDepth',[1 3],'Type','integer')
        optimizableVariable('numHiddenUnits1',[10 1000],'Type','integer')
        optimizableVariable('numHiddenUnits2',[10 1000],'Type','integer')
        optimizableVariable('numHiddenUnits3',[10 1000],'Type','integer')
        optimizableVariable('InitialLearnRate',[1e-3 1],'Transform','log')
        optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')
        optimizableVariable('miniBatchSize',[6 40],'Type','integer')
        optimizableVariable('LearnRateDropPeriod',[1 10],'Type','integer')
        optimizableVariable('LearnRateDropFactor',[0.1 0.9],'Type','real')
        optimizableVariable('dropoutrate',[0.1 0.7],'Type','real')
    ];
end

XValidation = XTest;
YValidation = YTest;


ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation,CNN_OR_GRU_OR_BiLSTM);
BayesObject = bayesopt(ObjFcn,optimVars, ...
    'MaxTime',14*60*60, ...
    'IsObjectiveDeterministic',false, ...
    'UseParallel',false,...
    'MaxObjectiveEvaluations',200);

bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError;

YPredicted = classify(savedStruct.trainedNet,XTest);
testError = 1 - mean(YPredicted == YTest);

end

function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation,CNN_OR_GRU_OR_BiLSTM)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        numClasses = numel(unique(YTrain));

        if CNN_OR_GRU_OR_BiLSTM == 1
            inputSize = [size(XTrain,1),1];
            numF = round(16/sqrt(optVars.SectionDepth));
            layers = [
                imageInputLayer(inputSize)
                convBlock(optVars.filterSize1,numF,optVars.SectionDepth)
                convBlock(optVars.filterSize2,2*numF,optVars.SectionDepth)
                convBlock(optVars.filterSize3,4*numF,optVars.SectionDepth)
                dropoutLayer(optVars.dropoutrate)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        elseif CNN_OR_GRU_OR_BiLSTM == 3
            inputSize = size(XTrain{1},1);
            layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmBlock(optVars.numHiddenUnits1,optVars.SectionDepth)
                bilstmBlock(optVars.numHiddenUnits2,optVars.SectionDepth)
                bilstmBlock(optVars.numHiddenUnits3,optVars.SectionDepth)
                dropoutLayer(optVars.dropoutrate)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        elseif CNN_OR_GRU_OR_BiLSTM == 2
            npaths = cellfun(@(x)size(x,1),XTrain);
            inputSize = npaths(1);
            layers = [ ...
                sequenceInputLayer(inputSize)
                gruBlock(optVars.numHiddenUnits1,optVars.SectionDepth)
                gruBlock(optVars.numHiddenUnits2,optVars.SectionDepth)
                gruBlock(optVars.numHiddenUnits3,optVars.SectionDepth)
                dropoutLayer(optVars.dropoutrate)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];   
        end

        miniBatchSize = optVars.miniBatchSize;
        validationFrequency = floor(numel(YTrain)/miniBatchSize);
        options = trainingOptions('adam', ...
            'InitialLearnRate',optVars.InitialLearnRate, ...
            'MaxEpochs',10, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',optVars.LearnRateDropPeriod, ...
            'LearnRateDropFactor',optVars.LearnRateDropFactor, ...
            'MiniBatchSize',miniBatchSize, ...
            'Shuffle','every-epoch', ...
            'Verbose',false, ...
            'Plots','none', ...
            'ValidationData',{XValidation,YValidation}, ...
            'ValidationFrequency',validationFrequency);

        trainedNet = trainNetwork(XTrain,YTrain,layers,options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
        YPredicted = classify(trainedNet,XValidation);
        valError = 1 - mean(YPredicted == YValidation);
        fileName = num2str(valError) + ".mat";
        save(fileName,'trainedNet','valError','options')
        cons = [];
        
    end
end


function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Padding','same')
    leakyReluLayer];
layers = repmat(layers,numConvLayers,1);
end

function layers = bilstmBlock(numHiddenUnits,numbilstmLayers)
layers = [
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    ];
layers = repmat(layers,numbilstmLayers,1);
end

function layers = gruBlock(numHiddenUnits,numgruLayers)
layers = [
    gruLayer(numHiddenUnits,'OutputMode','last')
    ];
layers = repmat(layers,numgruLayers,1);
end
