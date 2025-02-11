function [XGeneratedNew, TNew] = attributeTransfer(encoderNet, decoderNet, xto, yto, xfrom, yfrom)
% ANNOTATION:
% INPUT variables
% encoderNet —— the trained encoder
% decoderNet —— the trained decoder
% xto_subset —— the sample-scare area input data, it is 4-D format, each row is each feature, each colum is each sample
% yto —— the sample-scare area output data, it is categories
% xfrom_subset —— the sample-abundant area input data, it is 4-D format, each row is each feature, each colum is each sample
% yfrom —— the sample-abundant area output data, it is categories
% OUTPUT variables
% XGeneratedNew —— reconstructed data features
% TNew —— reconstructed data label

XGeneratedNew2 = [];
TNew2 = [];

yto = double(yto);
yfrom = double(yfrom);

for i = min(min([yto,yfrom])):max(max([yto,yfrom]))
labelsto = yto;

labelsfrom = yfrom;

idx2 = find(labelsto==i);

labelsto = labelsto(idx2);

xto_subset = xto(:,:,:,idx2);

idx2 = find(labelsfrom==i);

labelsfrom = labelsfrom(idx2);

xfrom_subset = xfrom(:,:,:,idx2);


Yto = categorical(labelsto);
Ytodl= permute(Yto,[2 3 4 1]);
Ytodl = double(string(Ytodl));
Ytodl=  dlarray(single(Ytodl), 'SSCB');
Ytodl = dlarray(reshape(single(Ytodl),[1,1,1,size(Ytodl,1)]), 'SSCB');
Xtodl = dlarray(single(xto_subset), 'SSCB');


Yfrom = categorical(labelsfrom);
Yfromdl= permute(Yfrom,[2 3 4 1]);
Yfromdl = double(string(Yfromdl));
Yfromdl=  dlarray(single(Yfromdl), 'SSCB');
Yfromdl = dlarray(reshape(single(Yfromdl),[1,1,1,size(Yfromdl,1)]), 'SSCB');
Xfromdl = dlarray(single(xfrom_subset), 'SSCB');
[z, ~, ~] = sampling_AttributeVector(encoderNet,Xfromdl,Yfromdl,Xtodl,Ytodl);

xPred = forward(decoderNet,z,Yfromdl);

XGeneratedNew = zeros(size(xto_subset,1), size(xPred,4));


TNew = zeros(1,1,1,size(Yfromdl,4));
for n = 1:size(xPred,4)
    XGeneratedNew(:,n) = xPred(:,:,1,n);
    TNew(1,1,1,n) = Yfromdl(1,1,1,n);
end




XGeneratedNew2 = [XGeneratedNew2, XGeneratedNew];
TNew2 = cat(4,TNew2,TNew);

end

XGeneratedNew = XGeneratedNew2;
TNew = TNew2;


end

function [zSampled, zMean, zLogvar] = sampling_AttributeVector(encoderNet,x,y,x2,y2)
    compressed2 = forward(encoderNet,x2,y2);
    compressed = forward(encoderNet,x,y);
    compressed = compressed + mean(compressed2,2) - mean(compressed,2);
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
