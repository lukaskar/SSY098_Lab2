function [w, w0] = process_epoch_not_shuffled(w, w0, lrate, examples_train, labels_train)

train_length = length(examples_train);

% One epoch of stochatsic gradient decent
for i = 1:train_length
    example = examples_train{i};
    label = labels_train(i);

    [wgrad, w0grad] = partial_gradient(w, w0, example, label);
    
    % Calculate new parameters according to gradient decent
    w = w - lrate * wgrad;
    w0 = w0 - lrate * w0grad;
end

end