%% General
paramDL.verbose = false;

%% Iterations
paramDL.numThreads=8; % number of threads
paramDL.iter=-5;  % let us see what happens after 100 iterations.

%% Learning mode
paramDL.mode=2;

%% Dictionary (components)
paramDL.modeD = 0;

if paramDL.modeD >= 1
    paramDL.gamma1 = 1.5;
    if paramDL.modeD == 2
        paramDL.gamma2 = 0.5;    
    end   
end

%% Positive constraints 
paramDL.posAlpha = 1; % positive contraints on the coefficients of the dictionary
paramDL.posD = 1; % positivity constraints on the dictionry 

paramDL.lambda = 1;
paramDL.lambda2 = 1;





