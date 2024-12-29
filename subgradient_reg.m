%% Load dataset
load('mds_train.mat')
functions = common_functions;

%% Define variables
n = 5;
x = -1/(n+sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
D = time_matrix.^2;

lambda = 0.9; % Regularization factor
alpha = 0.1; % Learning rate
max_iter = 100000; % Max iterations

H_original = eye(4,4);
G = V*H_original*V';
X_original = functions.get_X_from_XX(G);
[Dt,X_original] = procrustes(coords, X_original');
error_original = norm(coords - X_original, 'fro');

%% Function call
[H, error_vals, k_vals] = sgd(H_original, D, V, lambda, alpha, max_iter, coords);
G = V*H*V';
X_descent = functions.get_X_from_XX(G);
[Dt,X_descent] = procrustes(coords, X_descent');
error_descent = norm(coords - X_descent, 'fro');

%% Plot Convergence
figure(1)
plot(k_vals, error_vals)
xlabel('Iterations')
ylabel('Error')
title('Convergence of SGD with Regularization')

%% Plot True and Estimated Coordinates on a Map of the Netherlands
functions.plot_locations(coords, station_index, X_descent, X_descent, ...
    "SGD Estimated Locations", "SGD Estimated Locations")

%% Define functions

function subgradient = get_subgradient(H, D, V, lambda, regularization_type)
    n = size(D,1);
    e = ones(n,1);
    edm = diag(V*H*V')*e' + e*diag(V*H*V')' - 2*V*H*V';
    E = ones(n,n);

    % Core subgradient for reconstruction error
    error_subgrad = V'*(lambda*(edm - D)/norm(edm - D, 'fro'))*(2*E - 2*eye(n,n))*V;
    
    % Regularization subgradient
    if strcmp(regularization_type, 'frobenius')
        reg_subgrad = 2 * H; % Frobenius norm regularization gradient
    elseif strcmp(regularization_type, 'l1')
        reg_subgrad = sign(H); % L1 norm regularization subgradient
    else
        reg_subgrad = zeros(size(H)); % Default: no regularization
    end
    
    % Combine gradients
    subgradient = eye(n-1, n-1) + error_subgrad + lambda * reg_subgrad;
end

function [H, error_vals, k_vals] = sgd(H, D, V, lambda, alpha, max_iter, coords)
    functions = common_functions;
    error_vals = [];
    k_vals = [];
    regularization_type = 'forbenius'; % Choose 'frobenius' or 'l1'
    
    for k = 1:max_iter
        % Compute subgradient
        g = get_subgradient(H, D, V, lambda, regularization_type);
        
        % Update H with decreasing learning rate
        Hnew = H - (alpha/k) * g;
        
        % Eigenvalue decomposition to project onto PSD cone
        [Q, L] = eig(Hnew);
        L = max(L, 0); % Zero out negative eigenvalues
        H_psd = Q * L * Q';
        H = H_psd;
        
        % Record error every 1000 steps
        if (mod(k, 1000) == 1)
            G = V*H*V';
            X_descent = functions.get_X_from_XX(G);
            [Dt,X_descent] = procrustes(coords, X_descent');
            error_vals = [error_vals, norm(coords - X_descent, 'fro')];
            k_vals = [k_vals, k];
        end
    end
end
