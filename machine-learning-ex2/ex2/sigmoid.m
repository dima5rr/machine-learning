function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

    function g1 = sigmoid1(z1)
     g1 = 1/(1+1/(exp(z1)));    
    end

g = arrayfun(@sigmoid1, z);

% =============================================================

end
