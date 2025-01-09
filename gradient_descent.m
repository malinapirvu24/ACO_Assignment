%% Load dataset
load('mds_train.mat')

%% Define variables
n = 5;
x = -1/(n+sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
D = time_matrix.^2;

lambda = 0.2;
alpha = 0.2;
max_iter = 100;

H_original = eye(4,4);
G = V*H_original*V';
X_original = functions.get_X_from_XX(G);
[Dt,X_original] = procrustes(coords, X_original');
error_original = norm(coords - X_original, 'fro');

%% Function call
[H, function_vals, k_vals] = gradient_descent(H_original, D, V, lambda, alpha, max_iter);
G = V*H*V';
X_descent = functions.get_X_from_XX(G);
[Dt,X_descent] = procrustes(coords, X_descent');
error_descent = norm(coords - X_descent, 'fro');

% Plot True and Estimated Coordinates on a Map of the Netherlands
common_functions.plot_locations_general(coords, station_index, X_descent, "Gradient Descent Estimated Locations")

figure(3)
plot(k_vals, function_vals)

%% Plot error for different lambda
figure(2)
hold on
lambda_vals = linspace(0.1, 1, 10);
alpha_vals = [0.05, 0.1, 0.2];

for a = 1:length(alpha_vals)
    error_vals = [];
    alpha = alpha_vals(a); 
    for i = 1:length(lambda_vals)
        lambda_vals(i)
        [H, function_vals, k_vals] = gradient_descent(H_original, D, V, lambda_vals(i), alpha, max_iter);
        G = V*H*V';
        X_descent = functions.get_X_from_XX(G);
        [Dt,X_descent] = procrustes(coords, X_descent');
        error_val = norm(coords - X_descent, 'fro');
        error_val
        error_vals = [error_vals, error_val];
    end
    figure(3)
    hold on
    plot(lambda_values, error_vals, 'o-', 'LineWidth', 2, 'DisplayName', sprintf('\\alpha = %0d', alpha));
    xlabel('\lambda (Regularization Factor)');
    ylabel('Reconstruction Error (Frobenius Norm)');
    title('Error vs. \lambda');
    legend
    ax = gca; 
    ax.FontSize = 16; 
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

function [H, function_vals, k_vals] = gradient_descent(H, D, V, lambda, alpha, max_iter)
    function_vals = [];
    k_vals = [];
    
    n = size(D,1);
    e = ones(n,1);
    G = V*H*V';
    edm = diag(G)*e' + e*diag(G)' - 2*G;
    min_value = trace(H) + lambda*norm((edm-D), 'fro');
    min_H = H;

    for k = 1:max_iter
        g = get_subgradient(H, D, V, lambda);
        Hnew = H - (alpha/k)*g;
        % Eigenvalue decomposition
        [Q, L] = eig(Hnew);  
        % Zero-out negative eigenvalues
        L = max(L, 0); 
        % Project H onto the set of positive semi definite matrices
        Hnew_psd = Q * L * Q';
        
        % Keep track of minimum value
        G = V*Hnew_psd*V';
        edm = diag(G)*e' + e*diag(G)' - 2*G;
        function_val = trace(Hnew_psd) + lambda*norm((edm-D), 'fro');
        if (function_val < min_value)
            min_value = function_val;
            min_H = Hnew_psd;
        end

        H = Hnew_psd;

        % Get the error every 100 time steps for plotting
        if (mod(k, 2) == 1)
            k_vals = [k_vals, k];
            function_vals = [function_vals, function_val];
        end

    end
    H = min_H;
end
