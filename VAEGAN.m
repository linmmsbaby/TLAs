function [encoderNet,decoderNet,Discriminator,Lossinfo] = VAEGAN(XTrain,YTrain,encoderLG,decoderLG,Discriminator)
% Arrangement, modification and annotation by Mansheng Lin
% ANNOTATION:
% INPUT varibles
% XTrain —— it is 4-D format, each row is each feature, each colum is each sample
% YTrain —— it is the categories
% encoderLG —— for example:
% encoderLG = layerGraph([
%                         imageInputLayer([size(XTrain,1) 1 1],'Normalization','none','Name','in')
%                         concatenationLayer(3,2,'Name','cat')
%                         convolution2dLayer([6 1],16,'Stride',1,'Padding','same','Name','conv1')
%                         leakyReluLayer(0.2,'Name','lrelu1')
%                         convolution2dLayer([1 1],32,'Stride',1,'Padding','same','Name','conv2')
%                         leakyReluLayer(0.2,'Name','lrelu2')
%                         convolution2dLayer([4 1],64,'Stride',1,'Padding','same','Name','conv3')
%                         leakyReluLayer(0.2,'Name','lrelu3')
%                         dropoutLayer
%                         fullyConnectedLayer(2 * 200, 'Name', 'fc_encoder')]);
% decoderLG —— for example:
% decoderLG = layerGraph([
%     imageInputLayer([1 1 200],'Normalization','none','Name','in')
%     projectAndReshapeLayer([4 1 1024],200,'proj')
%     concatenationLayer(3,2,'Name','cat');
%     transposedConv2dLayer([2 1],8*64,'Name','tconv1')
%     batchNormalizationLayer('Name','bn1','Epsilon',5e-5)
%     reluLayer('Name','relu1')
%     transposedConv2dLayer([2 1],4*64,'Stride',1,'Name','tconv2')
%     batchNormalizationLayer('Name','bn2','Epsilon',5e-5)
%     reluLayer('Name','relu2')
%     transposedConv2dLayer([2 1],64,'Stride',1,'Name','tconv3')
%     batchNormalizationLayer('Name','bn3','Epsilon',5e-5)
%     reluLayer('Name','relu3')
%     transposedConv2dLayer([2 1],1,'Stride',2,'Name','tconv4')
%     ]);
% Discriminator —— for example:
% Discriminator = layerGraph([ 
%                         imageInputLayer([size(XTrain,1) 1 1],'Normalization','none','Name','in')
%                         concatenationLayer(3,2,'Name','cat')
%                         convolution2dLayer([6 1],16,'Stride',1,'Padding','same','Name','conv1')
%                         reluLayer
%                         convolution2dLayer([1 1],32,'Stride',1,'Padding','same','Name','conv2')
%                         reluLayer
%                         convolution2dLayer([4 1],64,'Stride',1,'Padding','same','Name','conv3')
%                         reluLayer
%                         dropoutLayer
%                         fullyConnectedLayer(1, 'Name', 'fc')]);
% OUTPUT varibles
% encoderNet —— trained encoder
% decoderNet —— trained decoder
% Discriminator —— trained discriminator
% Lossinfo —— elbo loss in each epoch


numTrain = size(XTrain,4);
numClasses = max(size(unique(double(YTrain))));
inputSize = [size(XTrain,1) 1 1];

% Construct Network
% Define an encoder and discriminator
embeddingDimension = 50;
encoderLabel = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(inputSize,embeddingDimension,numClasses,'emb')];
lgraphDiscriminator = addLayers(encoderLG,encoderLabel);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,'emb','cat/in2');
lgraphD = addLayers(Discriminator,encoderLabel);
lgraphD = connectLayers(lgraphD,'emb','cat/in2');

% Define an decoder 
projectionSize = [4 1 1024];

embeddingDimension = 20;
layers = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(projectionSize(1:2),embeddingDimension,numClasses,'emb')];
lgraphGenerator = addLayers(decoderLG,layers);
lgraphGenerator = connectLayers(lgraphGenerator,'emb','cat/in2'); 
encoderNet = dlnetwork(lgraphDiscriminator);
decoderNet = dlnetwork(lgraphGenerator);
Discriminator = dlnetwork(lgraphD);
% Specify Training Options
executionEnvironment = "auto";
numEpochs = 100;
miniBatchSize = 10;
lr = 2e-4;
numIterations = floor(numTrain/miniBatchSize);
iteration = 0; 
avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];
avgGradientsDiscriminator = [];
avgGradientsSquaredDiscriminator = [];
infoelbo = [];
infold = [];
YTraindl= permute(YTrain,[2 3 4 1]);
YTraindl=  dlarray(single(YTraindl), 'SSCB');
YTraindl = dlarray(reshape(single(YTraindl),[1,1,1,size(YTraindl,1)]), 'SSCB');
XTraindl = dlarray(single(XTrain), 'SSCB');
f = figure;
f.Position(3) = 2*f.Position(3);
scoreAxes = subplot(1,2,1);
lineLossTrain = animatedline(scoreAxes);
xlabel("Total Iterations")
ylabel("Loss")
scoreAxes = subplot(1,2,2);
lineLossTrain_D = animatedline(scoreAxes);
xlabel("Total Iterations")
ylabel("ClassiferLoss")
% Train Model
start = tic;
for epoch = 1:numEpochs
    tic;
    for i = 1:numIterations 
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize; 
        XBatch = XTrain(:,:,:,idx); 
        XBatch = dlarray(single(XBatch), 'SSCB'); 
        YBatch = permute(YTrain(idx),[2 3 4 1]);
        YBatch = dlarray(reshape(single(YBatch),[1,1,1,size(YBatch,1)]), 'SSCB');
 
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);           
        end 
        

        [infGrad, genGrad  ,genGrad_D,loss_Discriminator,loss] = dlfeval(...
            @modelGradients, encoderNet, decoderNet,Discriminator, XBatch,YBatch);
        

        [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderNet.Learnables, ...
                genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
                infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);


        [Discriminator, avgGradientsDiscriminator, avgGradientsSquaredDiscriminator] = ...
            adamupdate(Discriminator, ...
                genGrad_D, avgGradientsDiscriminator, avgGradientsSquaredDiscriminator, iteration, lr);
       

    end
    elapsedTime = toc;
    
    elbo = loss;
    infoelbo = [infoelbo,double(gather(extractdata(elbo)))];
    subplot(1,2,1)
    addpoints(lineLossTrain,iteration,double(gather(extractdata(elbo))))
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Loss - " + double(gather(extractdata(elbo))) + "; Epoch - " + epoch + "; Iteration - " + iteration + "; Time - " + string(D))
    drawnow   

    infold = [infold,double(gather(extractdata(loss_Discriminator)))];
    subplot(1,2,2)
    addpoints(lineLossTrain_D,iteration,double(loss_Discriminator))
    title("ClassiferLoss - " + double(gather(extractdata(loss_Discriminator))) + "; Epoch - " + epoch + "; Iteration - " + iteration + "; Time - " + string(D))
    drawnow   

end
Lossinfo.elboLoss = infoelbo;
Lossinfo.dLoss = infold;



end

% Functions
% modelGradients Functions

function [infGrad, genGrad,genGrad_D,loss_Discriminator,loss] = modelGradients(encoderNet, decoderNet, Discriminator, x,y)
    [z, zMean, zLogvar,epsilon] = sampling(encoderNet, x,y);
    xPred = forward(decoderNet, z,y);
    loss = ELBOloss(x, xPred, zMean, zLogvar);
    zP = dlarray(reshape(epsilon, [1,1,size(zMean)]), 'SSCB');
    xP = forward(decoderNet, zP,y);
    probReal = sigmoid(forward(Discriminator,x,y));
    probGen = sigmoid(forward(Discriminator,xPred,y));
    probGen2 = sigmoid(forward(Discriminator,xP,y));
    loss_Discriminator = -mean(log(probReal))-mean(log(1 - probGen))-mean(log(1 - probGen2));
    loss = loss + loss_Discriminator;
    [genGrad, infGrad] = dlgradient(loss, decoderNet.Learnables, ...
        encoderNet.Learnables);
    genGrad_D = dlgradient(loss_Discriminator, Discriminator.Learnables);
end

%Sampling and Loss Functions
function [zSampled, zMean, zLogvar,epsilon] = sampling(encoderNet,x,y)
    compressed = forward(encoderNet,x,y);
    d = size(compressed,1)/2;
    zMean = compressed(1:d,:);
    zLogvar = compressed(1+d:end,:);
    sz = size(zMean);
    epsilon = randn(sz);
    sigma = exp(.5 * zLogvar);
    z = epsilon .* sigma + zMean;
    z = reshape(z, [1,1,sz]);
    zSampled = dlarray(z, 'SSCB');
end

% The ELBOloss function
function elbo = ELBOloss(x, xPred, zMean, zLogvar)
    squares = 0.5*(xPred-x).^2;
    reconstructionLoss  = sum(squares, [1,2,3]);
    KL = -.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
    elbo = mean(reconstructionLoss + KL);
end

