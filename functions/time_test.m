function i = time_test(filter_size, num_filters, layer_size)
    for i = 1:layer_size(1) * layer_size(2) * layer_size(3)
        weights = rand(filter_size, filter_size, num_filters);
        sum(weights .* weights, 'all');
    end
i = 1;
end