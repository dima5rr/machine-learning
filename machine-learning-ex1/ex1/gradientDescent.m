function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
t0h =  zeros(num_iters, 1);
t1h =  zeros(num_iters, 1);

x = X(:,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    t0h(iter) = theta(1);
    t1h(iter) = theta(2);
    
    t0 = theta(1,1) - alpha*sum(X*theta-y)/m;
    t1 = theta(2,1) - alpha*((X*theta-y)'*x)/m;
    theta = [t0;t1];           



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

%plot(t0h,t1h)

end
