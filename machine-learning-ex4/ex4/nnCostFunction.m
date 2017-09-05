function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% h = sigmoid(Theta1*a1);
% J = (-y'*log(h) - (1-y)'*log(1-h))/m;

% X with bias unit
a1 = [ones(m, 1) X];

% A1,A2,..An should have same nunmber of features (columns) as X
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2_with_bias = [ones(size(a2,1), 1) a2];

z3 = a2_with_bias*Theta2';
a3 = sigmoid(z3);
hx = a3;

J = 0;

% map label matrix to 1s matrix
yy = zeros(size(a3));
for i=1:m
    label = y(i);
    yy(i,label) = 1;
end

% feedforward

% cost
cost = sum((-yy).*log(a3) - (1-yy).*log(1-a3), 2);
J = sum(cost)/m;

% regularized cost
% explicitly exclude the bias term θ0
theta1_wo_bias = Theta1(:,2:end);
theta2_wo_bias = Theta2(:,2:end);

p = (sum(sum(theta1_wo_bias.^2, 2)) + sum(sum(theta2_wo_bias.^2, 2)));
regularize_term = lambda * p/(2*m);
J = J + regularize_term;


% -------------------------------------------------------------
% backpropagation
% 1)
% feedforward
% a1 a2 a3 4

% 2)
% For each output unit k in layer 3 (the output layer)
d3 = a3-yy;

% 3)
% For the hidden layer l = 2
d2_1 = d3*Theta2;
d2 = d2_1(:,2:end).*sigmoidGradient(z2);

% 4)
delta1 = d2'*a1;
delta2 = d3'*a2_with_bias;

% 5)
% regularization (without bias term)
l1 = (lambda*Theta1(:,2:end))/m;
delta1 = delta1/m + [zeros(size(l1,1), 1) l1];

l2 = (lambda*Theta2(:,2:end))/m;
delta2 = delta2/m + [zeros(size(l2,1), 1) l2];

% In order to use optimizing functions such as "fminunc()",
% we need to "unroll" all the elements and put them into one long vector
grad = [reshape(delta1,[], 1); reshape(delta2, [], 1)];

% =========================================================================


end
