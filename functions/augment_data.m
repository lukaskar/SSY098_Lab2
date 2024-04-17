function [examples_train_aug,labels_train_aug] = augment_data(examples_train,labels_train,M)
examples_length = length(examples_train);   % Number of example images

% Initialize return variables
examples_train_aug = cell(1,examples_length * M);
labels_train_aug = zeros(1,examples_length * M);

for i = 1:M
    for j = 1:examples_length
        % Apply a random rotation and add to return variables
        index = (i - 1) * examples_length + j;
        rotation_angle = randi([0,3]) * 90; % 0, 90, 180 or 270 degrees

        examples_train_aug{index} = ...
            imrotate(examples_train{j}, rotation_angle);

        labels_train_aug(index) = labels_train(j);
    end
end
end