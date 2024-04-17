function net = train_classifier(layers, imgs_train, labels_train, ...
    imgs_val, labels_val)

% Train network
options = trainingOptions('sgdm', MaxEpochs=20);
net = trainNetwork(imgs_train, labels_train, layers, options);

% Validate network
val_length = length(imgs_val);  % Number of validation images
predicted_labels = categorical(zeros(val_length, 1));
for i = 1:val_length
    predicted_labels(i, 1) = net.classify(imgs_val(:,:,:,i));
end
correct = sum(predicted_labels==labels_val);
accuracy = 100*correct/val_length;
fprintf(("Validation Accuracy: ")+accuracy+("%%"))

end
