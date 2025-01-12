%% Load dataset
load('mds_train.mat');
functions = common_functions;

%% Define variables
n = 5;
x = -1/(n+sqrt(n));
y = -1/sqrt(n);
D = time_matrix.^2;

lambda = .4;
alpha = 1;
max_iter = 10000;
threshold = 1e-10;
threshold2 = 5;

G_original = eye(n,n);
X_original = functions.get_X_from_XX(G_original);
[Dt,X_original] = procrustes(coords, X_original');
error_original = norm(coords - X_original, 'fro');

%% Function call
tic
[G, function_vals, k_vals, k, converged] = gd(G_original, D, lambda, alpha, max_iter, threshold, threshold2);
toc
X_descent = functions.get_X_from_XX(G);
[Dt,X_descent] = procrustes(coords, X_descent');
error_descent = norm(coords - X_descent, 'fro');

% Plot True and Estimated Coordinates on a Map of the Netherlands
common_functions.plot_locations_general(coords, station_index, X_descent, "PDG Estimate", error_descent)

figure(2)
plot(k_vals, function_vals)

%% Plot error for different lambda
f1 = figure;
f2 = figure;
hold on
lambda_vals = linspace(0.1, 1, 10);
alpha_vals = [0.1];

for a = 1:length(alpha_vals)
    error_vals = [];
    ks = [];
    alpha = alpha_vals(a); 
    for i = 1:length(lambda_vals)
        [G, function_vals, k_vals, k, converged] = gd(G_original, D, lambda_vals(i), alpha, max_iter, threshold, threshold2);
        X_descent = functions.get_X_from_XX(G);
        [Dt,X_descent] = procrustes(coords, X_descent');
        error_val = norm(coords - X_descent, 'fro');
        error_vals = [error_vals, error_val];
        ks = [ks, k];
        sprintf("a = %0d, l = %0d, k = %0d, converged = %0d", alpha_vals(a), lambda_vals(i), k, converged)
    end
    figure(f1)
    hold on
    plot(lambda_vals, error_vals, 'o-', 'LineWidth', 2, 'DisplayName', sprintf('\\alpha = %0d', alpha));
    xlabel('\lambda (Regularization Factor)');
    ylabel('Reconstruction Error (Frobenius Norm)');
    title('Error vs. \lambda');
    legend
    ax = gca; 
    ax.FontSize = 16; 

    figure(f2)
    hold on
    plot(lambda_vals, ks, 'o-', 'LineWidth', 2, 'DisplayName', sprintf('\\alpha = %0d', alpha));
    xlabel('\lambda (Regularization Factor)');
    ylabel('Number of iterations');
    title('Number of iterations vs. \lambda');
    legend
    ax = gca; 
    ax.FontSize = 16; 
end
hold off
legend show


%% Define functions
function gradient = calculate_gradient(G, D, lambda)
    n = size(D,1);
    epsilon = 1e-10;
    f1 = obj_function(G, D, lambda);
    gradient = zeros(n, n);

    for i = 1:n-1
        for j = 1:n-1
            delta = zeros(n, n);
            delta(i,j) = epsilon;
            f2 = obj_function(G+delta, D, lambda);
            gradient(i,j) = (f2-f1)/epsilon;
        end
    end
end

function f = obj_function(G, D, lambda)
    n = size(D,1);
    e = ones(n,1);
    edm = norm(D,2)*(diag(G)*e' + e*diag(G)' - 2*G);
    f = trace(G) + lambda*norm((edm-D), 'fro');
end

function gradient = get_gradient(G, D, lambda)
    n = size(D,1);
    e = ones(n,1);

    % Find the EDM and normalize it using the norm of the D matrix
    edm = diag(G)*e' + e*diag(G)' - 2*G;
    edm = norm(D,2).*edm;

    edm_partial = zeros(n,n,n,n);
    % Compute the derivative of the EDM function w.r.t G
    for i = 1:n
        for j = 1:n
            if (i == j)
                ei_vec = zeros(n,1);
                ei_vec(i,1) = 1;
                edm_partial(:,:,i,j) = zeros(n,n) + ei_vec*e' + e*ei_vec';
                edm_partial(i,j,i,j) = edm_partial(i,j,i,j) - 2;
            else
                edm_partial(:,:,i,j) = zeros(n,n);
                edm_partial(i,j,i,j) = -2;
            end
        end
    end
    % Apply the chain rule for the EDM function and the Frobenius norm
    dfrob = ((edm-D)/norm(edm-D,'fro'))';

    chain_rule_partial = zeros(n,n);
    for i = 1:n
        for j = 1:n
            chain_rule_partial(i,j) = trace(dfrob*edm_partial(:,:,i,j));
        end
    end
    % Combine everything 
    gradient = eye(n, n) + lambda*(chain_rule_partial);

end

function [G, function_vals, k_vals, k, converged] = gd(G, D, lambda, alpha, max_iter, threshold, threshold2)
    function_vals = [];
    k_vals = [];
    
    n = size(D,1);
    e = ones(n,1);
    edm = diag(G)*e' + e*diag(G)' - 2*G;
    function_val_start = norm(D,2)*(trace(G) + lambda*norm((edm-D), 'fro'));
    min_value = function_val_start;
    min_G = G;

    for k = 1:max_iter

        % Iterate based on step size and gradient
        g = get_gradient(G, D, lambda);
        Gnew = G - (alpha/k)*g;

        % Project onto PSD cone
        [Q, L] = eig(Gnew);  
        % Zero-out negative eigenvalues
        L = real(L);
        L = max(L, 0); 
        Gnew = Q * L * Q';

        % Constrain G1 = 0
        for i = 1:n
            mean_row = mean(Gnew(i,:));  
            Gnew(i,:) = Gnew(i,:) - mean_row;
        end

        delta = norm(G-Gnew);
        G = Gnew;

        edm = norm(D,2)*(diag(G)*e' + e*diag(G)' - 2*G);
        function_val = trace(G) + lambda*norm((edm-D), 'fro');
       
        if (delta < threshold)
            break; 
        end 
        
        if (function_val < min_value)
            min_value = function_val;
            min_G = G;
        end

        % Get the error every 2 time steps for plotting
        if (mod(k, 2) == 1)
            k_vals = [k_vals, k];
            function_vals = [function_vals, function_val];
        end

    end
    edm = norm(D,2)*(diag(G)*e' + e*diag(G)' - 2*G);
    function_val = trace(G) + lambda*norm((edm-D), 'fro');
    gradient_norm = norm(get_gradient(G, D, lambda), 2);
    G = min_G;
    converged = (gradient_norm < threshold2 && function_val_start > function_val && ~(max(abs(eig(G))) < 1e-15));

end
