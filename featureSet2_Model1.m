
%% Optimized SVM classifier (Feature set 2 / Model 1)

% Imput data driven "Optimized SVM" classifier on feature set 2 for PVR prediciton and test the proposed
% model with a 5 fold cross-validation


% Input Variables : 
% -inputData : table containing 7 columns of features + 1 output 


% Output Variables : 
% - Confmat : (2x2) Classification Confusion Matrix
% - Confusion Chart 



inputTable = inputData;
features = {'AGE','DURATION_OF_SYMPTOMS','INTRAOCULAR_PRESSURE','SUBTOTAL RD','MACULAR_STATUS', 'GIANT_TEAR','VITREOUS_HEMORRHAGE'};
features = inputTable(:, features);
output = inputTable.PVR;
X = features;
Y = output;

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
 clfSVM = fitcsvm(...
    Xtrain, ...
    Ytrain, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 1, ...
    'BoxConstraint', 149.6967315681057, ...
    'Standardize', true, ...
    'ClassNames', categorical({'NON'; 'OUI'}));
%   % Classification 
   Pred= predict(clfSVM,Xtest);
%   % Confusion Matrix
   con=confusionmat(Ytest,Pred);
%   % Cumulative Confusion Matrix
   confmat=confmat+con; 
end
   
confusionchart(confmat)
