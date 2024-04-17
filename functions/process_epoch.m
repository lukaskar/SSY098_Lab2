function [w, w0] = process_epoch(w, w0, lrate, examples_train, labels_train)

train_length = length(examples_train);

% Shuffle data
shuffled_indices= randperm(train_length);
shuffled_examples = cell(1, train_length);
shuffled_labels = zeros(1, train_length);
for i = 1:train_length
    index = shuffled_indices(i);
    shuffled_examples(i) = examples_train(index);
    shuffled_labels(i) = labels_train(index);
end

% One epoch of stochatsic gradient decent
for i = 1:train_length
    example = shuffled_examples{i};
    label = shuffled_labels(i);

    [wgrad, w0grad] = partial_gradient(w, w0, example, label);
    
    % Calculate new parameters according to gradient decent
    w = w - lrate * wgrad;
    w0 = w0 - lrate * w0grad;
end

end