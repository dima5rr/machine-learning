function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% nu = number of users
% nm = number of movies
% r(i,j)=1 if user j has rated movie i
% y(i,j)= rating given by user j to movie i (defined only if r(i,j)=1)

% accumulating the cost for user j and movie i only if R(i,j) = 1.
f = (X*Theta' - Y).^2 .* R;
J = sum(sum(f)) / 2;
regularization_theta = sum(sum(Theta.*Theta))*lambda/2;
regularization_x = sum(sum(X.*X))*lambda/2;
J = J+regularization_theta+regularization_x;

% gradient (j=1..n - sum users rated that movie)
% *R - to sum only rated movies
for k=1:num_features
    regularization_x = X(:,k)*lambda;
    X_grad(:,k) = ((X*Theta' - Y).* R)*Theta(:,k) + regularization_x;
end

% theta (i=1..n - sum movies rated by user)
% *R - to sum only rated movies
for k=1:num_features
    regularization_theta = Theta(:,k)*lambda;
    Theta_grad(:,k) = ((X*Theta' - Y).* R)'*X(:,k) + regularization_theta;    
end




% =============================================================

 grad = [X_grad(:); Theta_grad(:)];

end
