function predicted_labels = classify(examples,w,w0)

examples_length = length(examples);
predicted_labels = zeros(1,examples_length);

% Classify each example image
for i= 1:examples_length
    y = sum(examples{i} .* w, 'all') + w0;

    if y > 0    % Classified positive
        predicted_labels(i) = 1;
    else        % Classified negative
        predicted_labels(i) = 0;
    end
end

end