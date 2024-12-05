load('mds_train.mat')

D = time_matrix.^2;

% Some variables needed for the convex problem
n = 5;
x = -1/(n+sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1,n-1);x*ones(n-1)+eye(n-1)];
e = ones(n,1);

% Semi-definite problem - see EDM paper
cvx_begin sdp
    variable G(n-1, n-1) symmetric nonnegative
    B = V*G*V';
    E = diag(B)*e' + e*diag(B)' - 2*B;
    maximize trace(G) - norm(E-D, 'fro');
    subject to 
        G >= 0;
cvx_end

% Get the EDM from the solved optimization problem
EDM = diag(B)*e'+e*diag(B)'-2*B;

% Find matrix of coordinates from the EDM 
XX = -1/2 * (EDM - EDM(:, 1) * ones(1, n) - ones(n, 1) * EDM(1, :));
[U, S, V] = svd(XX);
S = S(:, 1:2);
X = sqrt(S')*V';

% Perform a procrustes rotation 
% Can we use the true coords here? probably not?
[Dt,Z] = procrustes(coords, X');

% Plot the true coordinated alongside the estimates coordinates
figure(1)
plot(coords(:,1), coords(:,2), "x", Z(:,1), Z(:,2), "o")