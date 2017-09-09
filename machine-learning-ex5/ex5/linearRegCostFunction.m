function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



theta1 = theta(1);
thetaN = theta(2:end);

% h = 1.0 ./ (1.0 + exp(-(X*theta)));
h = X*theta;
regularize_term = lambda * sum(thetaN.^2)/(2*m);
J = sum((h-y).^2)/(2*m) + regularize_term;

%regularized gradient descent for linear regression
X0 = X(:,1);
XN = X(:,2:end);

grad0 = ((h-y)'*X0)/m;

regN = (lambda * thetaN)/m;
gradN = (((h-y)'*XN)/m)' +regN;

grad = [grad0;gradN];









% =========================================================================

grad = grad(:);

end
