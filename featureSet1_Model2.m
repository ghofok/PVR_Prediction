%% Optimizable Naive Bayes classifier (Feature set 1 / Model 2)

% Imput data driven "Optimizable Naive Bayes" classifier on feature set 1 for PVR prediciton and test the proposed
% model with a 5 fold cross-validation


% Input Variables : 
% -inputData : table containing 8 columns of features + 1 output 


% Output Variables : 
% - Confmat : (2x2) Classification Confusion Matrix
% - Confusion Chart 



inputTable = inputData;
features = {'AGE','DURATION_OF_SYMPTOMS','INTRAOCULAR_PRESSURE','SUBTOTAL RD','MACULAR_STATUS', 'GIANT_TEAR','VITREOUS_HEMORRHAGE',"PRE_EXISTING_PVR"};
features = inputTable(:, features);
output = inputTable.PVR;
X = features;
Y = output;
isCategoricalPredictor = [false, false, true, true, true, false, true, true];


rng('default'); 
k=5;
fold=cvpartition(Y,'kfold',k); % Divid data into 5 folds
confmat=0;


for i=1:k
    
  trainInt=fold.training(i); 
  testInt=fold.test(i);
%   
  Xtrain=X(trainInt,:);
  Ytrain=Y(trainInt,:);
%   
  Xtest=X(testInt,:) ;
  Ytest=Y(testInt,:);
  
%Model training
distributionNames =  repmat({'Kernel'}, 1, length(isCategoricalPredictor));
distributionNames(isCategoricalPredictor) = {'mvmn'};
if any(strcmp(distributionNames,'Kernel'))
    classificationNaiveBayes = fitcnb(...
    Xtrain, ...
    Ytrain, ...
        'Kernel', 'Epanechnikov', ...
        'Support', 'Unbounded', ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical({'NON'; 'OUI'}));
else
    classificationNaiveBayes = fitcnb(...
    Xtrain, ...
    Ytrain, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical({'NON'; 'OUI'}));
end

%   % Classification 
   Pred= predict(clfSVM,Xtest);
%   % Confusion Matrix
   con=confusionmat(Ytest,Pred);
%   % Cumulative Confusion Matrix
   confmat=confmat+con; 
end
   
confusionchart(confmat)
