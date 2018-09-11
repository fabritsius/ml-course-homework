function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Find the dimentions of out input
[r c] = size(z);
% Calculate SIGMOID for each value in the input
for i = 1:r
    for j = 1:c
        denom = 1 + exp(-z(i, j));
        g(i, j) = 1 / denom;
    end
end

% =============================================================

end
