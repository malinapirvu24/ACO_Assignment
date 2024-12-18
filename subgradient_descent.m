%% Load dataset

load('mds_train.mat')
n = 5;
x = -1/(n+sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
D = distance.^2;
lambda = 0.29;
alpha = 10;
max_iter = 10000;
H = 1.0e+04 * ...
   [0.1120   -0.0582    0.4871   -0.2795; ...
   -0.0582    1.7653   -0.0413   -1.3011; ...
    0.4871   -0.0413    2.2870   -1.5672; ...
   -0.2795   -1.3011   -1.5672    2.1177];
H = eye(4,4);
H = optimize(H, D, V, lambda, alpha, max_iter);

function subgradient = get_subgradient(H, D, V, lambda)
    n = size(D,1);
    e = ones(n,1);
    edm = diag(V*H*V')*e' + e*diag(V*H*V')' - 2*V*H*V';
    E = ones(n,n);
    %eye(n-1,n-1) + 
    subgradient = V'*(lambda*(edm-D)/norm(edm-D,'fro'))*(2*E-2*eye(n,n))*V;
end

function H = optimize(H0, D, V, lambda, alpha, max_iter)
    H = H0;
    for k = 1:max_iter
        H
        g = get_subgradient(H, D, V, lambda)
        Hnew = H - alpha*g;
        % Eigenvalue decomposition
        [Q, L] = eig(Hnew);  
        % Zero-out negative eigenvalues
        L = max(L, 0); 
        % Project H onto the set of positive semi definite matrices
        H_psd = Q * L * Q';
        H = H_psd;
    end
end