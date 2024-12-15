%% Load dataset

load('mds_train.mat')

%% Define the distance/time matrix
D = time_matrix.^2;
n = size(D,1);
lambda = 1; 

%% Get estimated coordinates 

X_sdr = sdr(D, n, lambda);
X_mds = mds(D, n);

%% Perform Procrustes rotation

[Dt,X_mds] = procrustes(coords, X_mds');
[Dt,X_sdr] = procrustes(coords, X_sdr');

%% Plot True and Estimated Coordinates on a Map of the Netherlands

lon_true = coords(:, 1); % Longitude
lat_true = coords(:, 2); % Latitude
lon_mds = X_mds(:,1); lat_mds = X_mds(:,2);
lon_sdr = X_sdr(:,1); lat_sdr = X_sdr(:,2);


% Create geographic axes
figure;
gx = geoaxes; % Geographic axes
geobasemap('grayland'); 

% Plot the true locations
geoplot(lat_true, lon_true, 'rx', 'LineWidth', 2, 'MarkerSize', 8);
hold on;

% Plot the MDS and SDR estimated locations
geoplot(lat_mds, lon_mds, 'bo', 'LineWidth', 2, 'MarkerSize', 8);
geoplot(lat_sdr, lon_sdr, 'go', 'LineWidth', 2, 'MarkerSize', 8);

% Add labels for true locations
for i = 1:length(lat_true)
    text(lat_true(i), lon_true(i), station_index{i}, ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'FontSize', 8, 'Color', 'black');
end

% Add legend
legend(["True Locations", "MDS Estimated Locations", "SDR Estimated Locations"], ...
       'Location', 'bestoutside');

% Set geographic limits to focus on the Netherlands
geolimits([50.5 53.7], [3.3 7.5]);

title('True and Estimated Locations on the Map of the Netherlands');
hold off;


%% Justify choosing lambda

% Define a range of lambda values
lambda_values = linspace(0.1, 1, 20); 

% Call the function to evaluate and plot error
figure(2)
plot_error_vs_lambda(lambda_values, D, coords, n);


%% Define functions 

% Perform singular value decomposition
% eigenvalues of G are real, and in the PSD case, they are non-negative
% eigenvalues  = singular values
function X = get_X_from_XX(XX)
    [U, S, V] = svd(XX);
    S = S(:, 1:2);
    X = sqrt(S')*V';
end


% MDS algorithm
% Find matrix of coordinates from the EDM using multi-dimensional scaling
function X = mds(edm, n) 
    XX = -1/2 * (edm - edm(:, 1) * ones(1, n) - ones(n, 1) * edm(1, :)); 
    % centering operation is included in this formula
    X = get_X_from_XX(XX);
end

function X = mds_diff_approach(D, n)
    n = size(D, 1);
    J = eye(n) - ones(n)/n;                   % Centering matrix
    G = -0.5 * J * (D.^2) * J;                % Gram matrix
    X = get_X_from_XX(D);
end



% SDR problem
% Semi-definite relaxation problem to complete an EDM - see EDM paper
function X = sdr(D, n, lambda)
    % Some variables needed for the convex problem
    x = -1/(n+sqrt(n));
    y = -1/sqrt(n);
    V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
    e = ones(n,1);

    % Solve the SDR convex problem using CVX
    cvx_begin sdp
        variable H(n-1, n-1) symmetric
        G = V*H*V';
        edm = diag(G)*e' + e*diag(G)' - 2*G;
        maximize (trace(H) - lambda*norm((edm-D), 'fro'));
        subject to 
            H >= 0;
    cvx_end

    X = get_X_from_XX(G);
end


% Error vs. lambda
function plot_error_vs_lambda(lambda_values, D, coords, n)

    for i = 1:length(lambda_values)
        lambda = lambda_values(i);
        
        % Estimate coordinates using SDR
        X_sdr = sdr(D, n, lambda);
        
        % Compute the Procrustes transformation to align the estimated coordinates
        [~, X_sdr_aligned] = procrustes(coords, X_sdr');
        
        % Calculate the reconstruction error using true coordinates
        errors(i) = norm(coords - X_sdr_aligned, 'fro'); % Frobenius norm of the difference
    end
    
    % Plot the error 
    plot(lambda_values, errors, 'o-', 'LineWidth', 2);
    grid on;
    xlabel('\lambda (Regularization Factor)');
    ylabel('Reconstruction Error (Frobenius Norm)');
    title('Error vs. \lambda');
    legend('Reconstruction Error');

end
