classdef common_functions

    methods (Static)

        % Perform singular value decomposition
        % eigenvalues of G are real, and in the PSD case, they are non-negative
        % square root of the eigenvalues
        function X = get_X_from_XX(XX)
            [U, D] = eig(XX);
            lambda = diag(D);

            [lambda, idx] = sort(lambda, 'descend');
            U = U(:, idx);

            d = 2;
            lambda_d = lambda(1:d);
            U_d = U(:, 1:d);

            sqrt_lambda_d = sqrt(lambda_d);

            S_d = diag(sqrt_lambda_d);

            n = size(XX, 1);
            Z = zeros(d, n - d);

            X = [S_d, Z] * U';
        end

        %% General plotting function using varargin

        function plot_locations_general(coords, station_index, varargin)
            lon_true = coords(:, 1); % Longitude
            lat_true = coords(:, 2); % Latitude

            figure;
            gx = geoaxes; % Geographic axes
            geobasemap('grayland');

            % Plot the true locations
            geoplot(lat_true, lon_true, 'rx', 'LineWidth', 2, 'MarkerSize', 8);
            hold on;

            for i = 1:length(lat_true)
                text(lat_true(i), lon_true(i), station_index{i}, ...
                    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
                    'FontSize', 8, 'Color', 'black');
            end

            % Process varargin
            num_inputs = length(varargin);
            assert(mod(num_inputs, 3) == 0, 'Arguments must be in coordinate-label pairs.');

            colors = ['b', 'g', 'm', 'c', 'y', 'k']; % Extend as needed
            color_index = 1;

            labels = [];
            for i = 1:3:num_inputs
                coords_set = varargin{i};
                labels = [labels, string(varargin{i+1} + ", Error: " + num2str(varargin{i+2}))];

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

            legend_items = [{'True Locations'}, labels];
            legend(legend_items, 'Location', 'bestoutside');

            % Set geographic limits for the Netherlands
            geolimits([50.5 53.7], [3.3 7.5]);

            title('True and Estimated Locations');
            ax = gca; 
            ax.FontSize = 16; 
            hold off;

        end
    end
end