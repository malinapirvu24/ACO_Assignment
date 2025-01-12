%% Load dataset
load('mds_train.mat')
functions = common_functions;

%% Define the distance/time matrix
D = distance.^2;
n = size(D,1);
lambda = 0.4; 

%% Get estimated coordinates 
[G_sdr, X_cvx] = cvx_alg(D, n, lambda);
X_mds = mds(D, n);

%% Perform Procrustes rotation
[Dt,X_mds] = procrustes(coords, X_mds');
[Dt,X_cvx] = procrustes(coords, X_cvx');

%% Plot True and Estimated Coordinates on a Map of the Netherlands
label1 = "MDS Estimate";
label2 = "CVX Estimate";
error_mds = norm(coords - X_mds, 'fro');
error_cvx = norm(coords - X_cvx, 'fro');
common_functions.plot_locations_general(coords, station_index, X_mds, label1, error_mds, X_cvx, label2, error_cvx);

%% Justify choosing lambda

% Define a range of lambda values
lambda_values = linspace(0.1, 1, 10); 
G_lambda = zeros(n,n,size(lambda_values,2));
X_lambda = zeros(2,n,size(lambda_values,2));

for i = 1:length(lambda_values)
    lambda = lambda_values(i); 
    % Estimate coordinates using CVX
    [G_lambda(:,:,i), X_lambda(:,:,i)] = cvx_alg(D, n, lambda);
end

% Call the function to evaluate and plot error
figure(2)
plot_eigenvalues_vs_lambda(G_lambda, lambda_values);

figure(3)
plot_error_vs_lambda(X_lambda, coords, lambda_values);

%% Define functions 

% MDS algorithm
% Find matrix of coordinates from the EDM using multi-dimensional scaling
function X = mds(edm, n) 
    XX = -1/2 * (edm - edm(:, 1) * ones(1, n) - ones(n, 1) * edm(1, :)); 
    % centering operation is included in this formula
    X = common_functions.get_X_from_XX(XX);
end

% Semi-definite relaxation problem to complete an EDM - see EDM paper
function [G, X] = cvx_alg(D, n, lambda)
    % Some variables needed for the convex problem
    x = -1/(n+sqrt(n));
    y = -1/sqrt(n);
    V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
    e = ones(n,1);

    % Solve the convex problem using CVX
    cvx_begin sdp
        variable H(n-1, n-1) symmetric
        G = V*H*V';
        edm = diag(G)*e' + e*diag(G)' - 2*G;
        minimize (trace(H) + lambda*norm((edm-D), 'fro'));
        subject to 
            H >= 0;
    cvx_end

    X = common_functions.get_X_from_XX(G);
end


% Error vs. lambda
function plot_error_vs_lambda(X_lambda, coords, lambda_values)

    errors = zeros(size(lambda_values));
    for i = 1:length(lambda_values)
        % Compute the Procrustes transformation to align the estimated coordinates
        [~, X_aligned] = procrustes(coords, X_lambda(:,:,i)');
        % Calculate the reconstruction error using true coordinates
        errors(i) = norm(coords - X_aligned, 'fro'); % Frobenius norm of the difference
    end
    
    % Plot the error 
    plot(lambda_values, errors, 'o-', 'LineWidth', 2);
    grid on;
    xlabel('\lambda (Regularization Factor)');
    ylabel('Reconstruction Error (Frobenius Norm)');
    title('Error vs. \lambda');
    legend('Reconstruction Error');
    ax = gca; 
    ax.FontSize = 16; 
end

% Eigenvalues vs. lambda (log scale)
function plot_eigenvalues_vs_lambda(G_lambda, lambda_values)
    % Loop over lambda values
    hold on;
    for i = 1:length(lambda_values)
        lambda = lambda_values(i);

        % Compute eigenvalues of the Gram matrix
        eigenvalues = abs(eig(G_lambda(:,:,i)));
        
        % Sort eigenvalues in descending order
        eigenvalues_sorted = sort(eigenvalues, 'descend');
        normalize_eigenvalues_sorted = eigenvalues_sorted/(sum(eigenvalues_sorted));

        if (mod(i,2) == 0)
            line = 'o-.';
        else
            line = '^-';
        end

        % Plot eigenvalues on a log scale
        semilogy(1:length(normalize_eigenvalues_sorted), normalize_eigenvalues_sorted, line, 'LineWidth', 2, ...
                 'DisplayName', ['\lambda = ' num2str(lambda)]);
    end
    hold off;
    
    % Add grid, labels, and legend
    grid on;
    xlabel('Index of Eigenvalue');
    ylabel('Eigenvalues (absolute value, normalized, log scale)');
    title('Eigenvalues of Gram Matrix vs. \lambda');
    set(gca, 'YScale', 'log')
    legend('show');
    ax = gca; 
    ax.FontSize = 16; 
end
