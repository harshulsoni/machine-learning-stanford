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

hx=X*theta;
hx=hx-y;
h2=hx.^2;
h2=sum(h2);
theta(1)=0;
thetas=theta.^2;
thetas=sum(thetas)*lambda;
J=h2+thetas;
J=J/(2*m);


grad=X'*hx+lambda*theta;
grad=grad/m;









% =========================================================================

grad = grad(:);

end
