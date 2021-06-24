function [W, C] = nmf_3dmm(X,paramDL, paramLasso)
% Weights(?)
disp('Building dictionary...')
tic
[W] = mexTrainDL(X,paramDL);
toc
% Components
disp('Computing sparse components...')
tic
C=mexLasso(X,W,paramLasso);
toc
end