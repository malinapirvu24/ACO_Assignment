%% Load dataset

load('mds_train.mat')

%% Define variables
n = 5;
x = -1/(n+sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
D = time_matrix.^2;


lambda = 0.29;
alpha = 4;
max_iter = 10000;


H = eye(4,4);
H = 1.0e+04 * ...
   [0.1120   -0.0582    0.4871   -0.2795; ...
   -0.0582    1.7653   -0.0413   -1.3011; ...
    0.4871   -0.0413    2.2870   -1.5672; ...
   -0.2795   -1.3011   -1.5672    2.1177];


%% Function call

H = optimize(H, D, V, lambda, alpha, max_iter);
G = V*H*V';
X_descent = get_X_from_XX(G);
[Dt,X_descent] = procrustes(coords, X_descent');


%% Plot True and Estimated Coordinates on a Map of the Netherlands

lon_true = coords(:, 1); % Longitude
lat_true = coords(:, 2); % Latitude
lon_descent = real(X_descent(:,1)); lat_descent = real(X_descent(:,2));


% Create geographic axes
figure;
gx = geoaxes; % Geographic axes
geobasemap('grayland'); 

% Plot the true locations
geoplot((lat_true), lon_true, 'rx', 'LineWidth', 2, 'MarkerSize', 8);
hold on;

% Plot the descent estimated locations
geoplot(lat_descent, lon_descent, 'bo', 'LineWidth', 2, 'MarkerSize', 8);


% Add labels for true locations
for i = 1:length(lat_true)
    text(lat_true(i), lon_true(i), station_index{i}, ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'FontSize', 8, 'Color', 'black');
end

% Add legend
legend(["True Locations", "Descent Estimated Locations"], ...
       'Location', 'bestoutside');

% Set geographic limits to focus on the Netherlands
geolimits([50.5 53.7], [3.3 7.5]);

title('True and Estimated Locations on the Map of the Netherlands');
hold off;



%% Define functions

function subgradient = get_subgradient(H, D, V, lambda)
    n = size(D,1);
    e = ones(n,1);
    edm = diag(V*H*V')*e' + e*diag(V*H*V')' - 2*V*H*V';
    E = ones(n,n);
    subgradient = eye(n-1, n-1) + V'*(lambda*(edm-D)/norm(edm-D,'fro'))*(2*E-2*eye(n,n))*V;
end

function H = optimize(H, D, V, lambda, alpha, max_iter)
    for k = 1:max_iter
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

function X = get_X_from_XX(XX)
    [U, S, V] = svd(XX);
    S = S(:, 1:2);
    X = sqrt(S')*V';
end