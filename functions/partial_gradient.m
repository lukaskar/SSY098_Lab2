function [wgrad, w0grad] = partial_gradient(w, w0, example_train, label_train)

y = sum(example_train .* w, 'all') + w0;
p = exp(y) / (1 + exp(y));

if label_train == 1     % Positive example
    dLdp = -1 / p;
else                    % Negative example
    dLdp = 1 / (p - 1);
end