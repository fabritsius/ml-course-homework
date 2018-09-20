function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.03;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Set C and sigma for a loop (original C and sigma store best values)
C_curr = C;
sigma_curr = sigma;
% Set initial error to the biggest value
lowest_err = Inf;
% Loop through all values of C and sigma
for i = 1:8
    for j = 1:8
        % Train the model
        model = svmTrain(X, y, C_curr, @(x1, x2) gaussianKernel(x1, x2, sigma_curr));
        % Compute the prediction error
        pred_err = mean(double(svmPredict(model, Xval) ~= yval));
        % Check if error is the lowest yet
        if pred_err < lowest_err
            lowest_err = pred_err;
            C = C_curr;
            sigma = sigma_curr;
        end
        fprintf('%d%% C=%f sigma=%f Error=%f\n', 12.5*(i-1) + 1.56*j, ...
        C_curr, sigma_curr, pred_err);
        % Increase sigma
        % in MATLAB use round(sigma_curr * 3, 2)
        sigma_curr = round(sigma_curr * 333) / 100;
    end
    % Increase C and reset sigma
    % in MATLAB use round(C_curr * 3, 2)
    C_curr = round(C_curr * 333) / 100;
    sigma_curr = 0.03;
end

% =========================================================================

end
