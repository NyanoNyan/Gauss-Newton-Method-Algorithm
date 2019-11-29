# Guass-Newton-Method-Algorithm
Training Sunspot Data using Guass-newton method (using the Hessian matrix) - Time Series


![Screenshot](pic.JPG)


Guass-Newtons Method

The Gauass-Newtons Method converges very fast although it can be seen as unstable, the computation complexity for this method will only require the computation of Jacobian.

w_(k+1)=w_k-(J_k^T J_k )^(-1) J_k e_k


The Hessian can be approximately calculated by J T*J, where J is the Jacobian matrix. The Jacobian matrix consists of size M X W, where M  is the number of patterns and W is the number of weights in the model. It consists of the error of each pattern respect its weights. 

H≈J_k^T J_k

