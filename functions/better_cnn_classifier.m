function layers = better_cnn_classifier()
layers = [
    imageInputLayer([28 28 1]);
    convolution2dLayer(3, 64);
    reluLayer();
    convolution2dLayer(3, 64);
    reluLayer();
    maxPooling2dLayer(2, Stride=2);
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()];