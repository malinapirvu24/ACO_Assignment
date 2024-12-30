classdef common_functions

    methods (Static)

        %% Perform singular value decomposition
        % eigenvalues of G are real, and in the PSD case, they are non-negative
        % square root of the eigenvalues

        function X = get_X_from_XX(XX)
            % Perform Eigenvalue Decomposition (EVD)
            [U, D] = eig(XX);
            lambda = diag(D);

            % Sort eigenvalues and eigenvectors in descending order
            [lambda, idx] = sort(lambda, 'descend');
            U = U(:, idx);

            % Take the top 2 eigenvalues
            d = 2;
            lambda_d = lambda(1:d);
            U_d = U(:, 1:d);

            % Compute the square root of the eigenvalues
            sqrt_lambda_d = sqrt(lambda_d);

            % Create the diagonal matrix with square root of eigenvalues
            S_d = diag(sqrt_lambda_d);

            % Zero matrix for remaining dimensions
            n = size(XX, 1);
            Z = zeros(d, n - d);

            X = [S_d, Z] * U';
        end

        %% General plotting function using varargin

        function plot_locations_general(coords, station_index, varargin)
            lon_true = coords(:, 1); % Longitude
            lat_true = coords(:, 2); % Latitude

            % Create geographic axes
            figure;
            gx = geoaxes; % Geographic axes
            geobasemap('grayland');

            % Plot the true locations
            geoplot(lat_true, lon_true, 'rx', 'LineWidth', 2, 'MarkerSize', 8);
            hold on;

            % Add labels for true locations
            for i = 1:length(lat_true)
                text(lat_true(i), lon_true(i), station_index{i}, ...
                    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
                    'FontSize', 8, 'Color', 'black');
            end

            % Process varargin
            num_inputs = length(varargin);
            assert(mod(num_inputs, 2) == 0, 'Arguments must be in coordinate-label pairs.');

            % Define colors for the coordinate sets
            colors = ['b', 'g', 'm', 'c', 'y', 'k']; % Extend as needed
            color_index = 1;

            for i = 1:2:num_inputs
                % Extract coordinates and label
                coords_set = varargin{i};
                label = varargin{i+1};

                % Check if coordinates are Nx2
                assert(size(coords_set, 2) == 2, 'Coordinate sets must be Nx2 matrices.');

                % Extract longitude and latitude
                lon = coords_set(:, 1);
                lat = coords_set(:, 2);

                % Plot the coordinates
                geoplot(lat, lon, [colors(color_index), 'o'], ...
                    'LineWidth', 2, 'MarkerSize', 8);

                % Iterate colors
                color_index = mod(color_index, length(colors)) + 1;
            end

            % Add legend
            legend_items = [{'True Locations'}, varargin(2:2:end)];
            legend(legend_items, 'Location', 'bestoutside');

            % Set geographic limits for the Netherlands
            geolimits([50.5 53.7], [3.3 7.5]);

            title('True and Estimated Locations on the Map of the Netherlands');
            hold off;
        end
    end
end