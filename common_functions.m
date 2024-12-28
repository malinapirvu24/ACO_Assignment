classdef common_functions

methods (Static)

    % Perform singular value decomposition
    % eigenvalues of G are real, and in the PSD case, they are non-negative
    % eigenvalues  = singular values
    function X = get_X_from_XX(XX)
        [U, S, V] = svd(XX);
        S = S(:, 1:2);
        X = sqrt(S')*V';
    end

    function plot_locations(coords, station_index, X1, X2, X1_label, X2_label)
    lon_true = coords(:, 1); % Longitude
    lat_true = coords(:, 2); % Latitude
    lon_x1 = X1(:,1); lat_x1 = X1(:,2);
    lon_x2 = X2(:,1); lat_x2 = X2(:,2);
    
    % Create geographic axes
    figure;
    gx = geoaxes; % Geographic axes
    geobasemap('grayland'); 
    
    % Plot the true locations
    geoplot(lat_true, lon_true, 'rx', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    
    % Plot the MDS and SDR estimated locations
    geoplot(lat_x1, lon_x1, 'bo', 'LineWidth', 2, 'MarkerSize', 8);
    geoplot(lat_x2, lon_x2, 'go', 'LineWidth', 2, 'MarkerSize', 8);
    
    % Add labels for true locations
    for i = 1:length(lat_true)
        text(lat_true(i), lon_true(i), station_index{i}, ...
            'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
            'FontSize', 8, 'Color', 'black');
    end
    
    % Add legend
    legend(["True Locations", X1_label, X2_label], ...
           'Location', 'bestoutside');
    
    % Set geographic limits to focus on the Netherlands
    geolimits([50.5 53.7], [3.3 7.5]);
    
    title('True and Estimated Locations on the Map of the Netherlands');
    hold off;
end

end
end