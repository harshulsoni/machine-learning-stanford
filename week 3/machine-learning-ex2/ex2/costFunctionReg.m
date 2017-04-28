function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hx=X*theta;
hx=sigmoid(hx);
lhx=log(hx);
lhx1=log(1-hx);
J=(y.*lhx)+((1-y).*lhx1);
J=-1*J;
J=sum(sum(J));
theta(1)=0;
theta2=theta.^2;
J=J+(lambda)*sum(theta2)/2;
J=J/m;
hx=hx-y;
grad=X'*hx+lambda*theta;
grad=grad/m;








% =============================================================

end
