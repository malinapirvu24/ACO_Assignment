%% Load dataset
load('mds_train.mat')

%% Define variables
n = 5;
x = -1/(n+sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
D = time_matrix.^2;

lambda = 0.5;
alpha = 0.1;
max_iter = 100000;
threshold = 5e-6;

H_original = eye(4,4);
G = V*H_original*V';
X_original = functions.get_X_from_XX(G);
[Dt,X_original] = procrustes(coords, X_original');
error_original = norm(coords - X_original, 'fro');

%% Function call
[H, error_vals, k_vals] = sgd(H_original, D, V, lambda, alpha, max_iter, coords, threshold);
G = V*H*V';
X_descent = functions.get_X_from_XX(G);
[Dt,X_descent] = procrustes(coords, X_descent');
error_descent = norm(coords - X_descent, 'fro');

% Plot True and Estimated Coordinates on a Map of the Netherlands
common_functions.plot_locations(coords, station_index, X_descent, X_descent, "SGD Estimated Locations", "SGD Estimated Locations")

%% Plot error for different lambda
figure(2)
hold on
lambda_vals = linspace(0.1, 1, 10);
for i = 1:length(lambda_vals)
    [H, error_vals, k_vals] = sgd(H_original, D, V, lambda_vals(i), alpha, max_iter, coords, threshold);
    plot(k_vals, error_vals, 'DisplayName', ['\lambda = ' num2str(lambda_vals(i))]);
end
hold off
legend show


%% Define functions
function subgradient = get_subgradient(H, D, V, lambda)
    n = size(D,1);
    e = ones(n,1);
    edm = diag(V*H*V')*e' + e*diag(V*H*V')' - 2*V*H*V';
    E = ones(n,n);
    subgradient = eye(n-1, n-1) + V'*(lambda*(edm-D)/norm(edm-D,'fro'))*(2*E-2*eye(n,n))*V;
end

function [H, error_vals, k_vals] = sgd(H, D, V, lambda, alpha, max_iter, coords, threshold)
    error_vals = [];
    k_vals = [];
    
    for k = 1:max_iter
        g = get_subgradient(H, D, V, lambda);
        Hnew = H - (alpha/k)*g;
        % Eigenvalue decomposition
        [Q, L] = eig(Hnew);  
        % Zero-out negative eigenvalues
        L = max(L, 0); 
        % Project H onto the set of positive semi definite matrices
        Hnew_psd = Q * L * Q';
        
        % Stopping condition
        if (norm(H - Hnew_psd, 'fro') < threshold)
            return
        end
        H = Hnew_psd;

        % Get the error every 100 time steps for plotting
        if (mod(k, 100) == 1)
            G = V*H*V';
            X_descent = common_functions.get_X_from_XX(G);
            [Dt,X_descent] = procrustes(coords, X_descent');
            error_vals = [error_vals, norm(coords - X_descent, 'fro')];
            k_vals = [k_vals, k];
        end
    end
end
