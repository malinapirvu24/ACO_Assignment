load('mds_train.mat')

D = time_matrix.^2;

n = size(D,1);
W = ones(n, n);
lambda = 1; 

X_sdr = sdr(D, n, W, lambda);
X_mds = mds(D, n);

% Perform a procrustes rotation 
% Can we use the true coords here? probably not?
[Dt,X_mds] = procrustes(coords, X_mds');
[Dt,X_sdr] = procrustes(coords, X_sdr');

% Plot the true coordinated alongside the estimates coordinates
figure(1)
plot(coords(:,1), coords(:,2), "x", X_mds(:,1), X_mds(:,2), "o", X_sdr(:,1), X_sdr(:,2), "o")
legend(["True locations", "Estimated locations using MDS", "Estimated locations using SDR"])
% Label the points with the train station names
hold on;
for i = 1:length(coords)
    text(coords(i,1), coords(i,2), station_index{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
hold off;

% Find matrix of coordinates from the EDM using multi-dimensional scaling
function X = mds(edm, n)
    XX = -1/2 * (edm - edm(:, 1) * ones(1, n) - ones(n, 1) * edm(1, :));
    X = get_X_from_XX(XX);
end

function X = get_X_from_XX(XX)
    [U, S, V] = svd(XX);
    S = S(:, 1:2);
    X = sqrt(S')*V';
end

% Semi-definite problem to complete an EDM - see EDM paper
function X = sdr(D, n, W, lambda)
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
        maximize (trace(H) - lambda*norm(W.*(edm-D), 'fro'));
        subject to 
            H >= 0;
    cvx_end

    X = get_X_from_XX(G);

end