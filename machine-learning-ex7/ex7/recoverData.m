function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               

% Q = reshape([1:15],5,3);
% recoverData(Q, magic(5), 3)
% 
% % result
% ans =
%    172   130   183   291   394
%    214   165   206   332   448
%    256   200   229   373   502
%    298   235   252   414   556
%    340   270   275   455   610
   
u_reduce = U(:, 1:K);
X_rec = Z*u_reduce';

% =============================================================

end
