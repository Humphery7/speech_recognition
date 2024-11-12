% %% Load Data
% speechFolder = 'C:\Users\computer world\Documents\Matlab_Projects\data_commands.zip';
% % dataFolder = tempdir;
% unzip(speechFolder, "C:\Users\computer world\Documents\Matlab_Projects\dataFolder");
%% 
dataset = fullfile("C:\Users\computer world\Documents\Matlab_Projects\speechRecognition", "dataFolder");
%% 
trainFolderpath = fullfile(dataset, "train");
disp(['trainfolder: ', trainFolderpath])
% %% Create Training Data Store
ads = audioDatastore(trainFolderpath, IncludeSubfolders=true,LabelSource="foldernames",FileExtensions='.wav')
%commands = categorical(["TOL1","TOL2","TOL3","TOL4","TOFL1","TOFL2","TOFL3","TOFL4"]);
%background = categorical("background");
%% 
commands = categorical(["TOL1","TOL2","TOL3","TOL4","TOFL1","TOFL2","TOFL3","TOFL4"]);
isCommand = ismember(ads.Labels,commands);
% isUnknown = ~isCommand;
% 
% includeFraction = 0.2;
% mask = rand(numel(ads.Labels),1) < includeFraction;
% isUnknown = isUnknown & mask;
% ads.Labels(isUnknown) = categorical("unknown");
% ads.Labels(isUnknown)
%% 
adsTrain = subset(ads, isCommand);
countEachLabel(adsTrain)
%% 
fs = 48000;

segmentDuration = 3;
frameDuration = 0.025;
hopDuration = 0.01;

FFTLength = 2048;
numBands = 50;

segmentSamples = round(segmentDuration * fs);
frameSamples = round(frameDuration * fs);
hopSamples = round(hopDuration * fs);
overlapSamples = frameSamples - hopSamples;

afe = audioFeatureExtractor(SampleRate=fs, ...
    FFTLength=FFTLength, ...
    window=hann(frameSamples, "periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);

setExtractorParameters(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);
%% 
% x = read(adsTrain);
% if size(x,2) == 2
%     xMono = mean(x,2);
% end
% numSamples = size(xMono,1);
% 
% numToPadFront = floor((segmentSamples - numSamples)/2);
% numToPadBack = ceil((segmentSamples - numSamples)/2);
% 
% 
% 
% xPadded = [zeros(numToPadFront,1,'like',xMono); xMono; zeros(numToPadBack,1,'like',xMono)];
% 
% %% 
% features = extract(afe, xPadded);
% [numHops, numFeatures] = size(features)
%% 

% Number of samples to display
numSamplesToDisplay = 3;

% Loop through the first few audio samples in the datastore
for i = 1:numSamplesToDisplay
    % Read and pad each audio sample
    audio = read(adsTrain);  % Read the next audio in the datastore
    numSamples = size(audio, 1);
    
    % Padding calculation
    numToPadFront = floor((segmentSamples - numSamples) / 2);
    numToPadBack = ceil((segmentSamples - numSamples) / 2);
    
    % Pad the audio to match segment duration
    audioPadded = [zeros(numToPadFront, 2, 'like', audio); audio; zeros(numToPadBack, 2, 'like', audio)];
    
    % Convert to mono by averaging channels if stereo
    if size(audioPadded, 2) == 2
        audioPadded = mean(audioPadded, 2);
    end
    
    % Extract features (Bark spectrum)
    features = extract(afe, audioPadded);
    
    % Plot the audio waveform and spectrogram
    figure;
    
    % Plot audio waveform
    subplot(2,1,1);
    t = (0:length(audioPadded)-1) / fs;  % Time vector
    plot(t, audioPadded);
    title(['Audio Waveform - Sample ', num2str(i)]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    

    % Plot spectrogram
    subplot(2,1,2);
    imagesc(10*log10(features.'));  % Convert to dB scale for visualization
    axis xy;
    colormap jet;
    colorbar;
    title(['Bark Spectrum - Sample ', num2str(i)]);
    xlabel('Time Frames');
    ylabel('Frequency Bands');
end
%% 
% Define an audio feature extractor with a mel spectrogram
% afe = audioFeatureExtractor(SampleRate=fs, ...
%     FFTLength=FFTLength, ...
%     window=hann(frameSamples, "periodic"), ...
%     OverlapLength=overlapSamples, ...
%     linearSpectrum=true);
% 
% setExtractorParameters(afe,"linearSpectrum");
% 
% % Read the audio and prepare it
% x = read(adsTrain);
% if size(x,2) == 2
%     xMono = mean(x,2);  % Convert to mono if stereo
% end
% 
% % Pad or trim audio segment to match segment duration
% numSamples = size(xMono,1);
% numToPadFront = floor((segmentSamples - numSamples)/2);
% numToPadBack = ceil((segmentSamples - numSamples)/2);
% xPadded = [zeros(numToPadFront,1,'like',xMono); xMono; zeros(numToPadBack,1,'like',xMono)];
% 
% % Extract mel spectrogram features
% features = extract(afe, xPadded);
% 
% % Plot the mel spectrogram
% figure;
% melSpectrogram(xPadded, fs, 'FFTLength', FFTLength, 'NumBand', numBands, ...
%                'OverlapLength', overlapSamples);
% title('Mel Spectrogram');
% colorbar;
% xlabel('Time (s)');
% ylabel('Frequency (Hz)');
%% 
YTrain = adsTrain.Labels;

transform1 = transform(adsTrain,@(x)[zeros(floor((segmentSamples-size(x,1))/2),2);x;zeros(ceil((segmentSamples-size(x,1))/2),2)]);
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)})

XTrain = readall(transform3);


% %% 
% 
% % Specify the fraction of data to use for validation
% validationFraction = 0.2;  % 20% of data for validation
% 
% % Split the data (adsTrain) into training and validation sets
% [adsTrainSplit, adsValidationSplit] = splitEachLabel(adsTrain, 1 - validationFraction, 'randomized');
% 
% % Check the number of samples in each set
% disp("Training Set: " + numel(adsTrainSplit.Files) + " samples");
% disp("Validation Set: " + numel(adsValidationSplit.Files) + " samples");
% X = readall(adsTrainSplit);  % X now contains a cell array of images
% % Assuming your data is in X and labels in TTrain
% numSamples = size(X, 1);
% validationFraction = 0.2;  % 20% for validation
% numValidation = floor(numSamples * validationFraction);
% numTrain = numSamples - numValidation;
% 
% % Randomly permute the indices
% randIndices = randperm(numSamples);
% 
% % Split the data
% XTrain = X(randIndices(1:numTrain), :);  % Training data
% TTrainSplit = TTrain(randIndices(1:numTrain), :);  % Training labels
% XValidation = X(randIndices(numTrain+1:end), :);  % Validation data
% TValidationSplit = TTrain(randIndices(numTrain+1:end), :);  % Validation labels
% 
% % Check the sizes
% disp("Training data size: " + size(XTrain));
% disp("Validation data size: " + size(XValidation));

%% 
XTrain = cat(4, XTrain{:});  % This consolidates the data into a 4D matrix
XTrain = mean(XTrain, 3);  
[numHops,numBands,numChannels,numFiles] = size(XTrain)


classes = commands
labelCountsTable = countEachLabel(adsTrain)
classCounts = labelCountsTable.Count;
classCounts
%% 


classWeights = 1 ./ classCounts;
classWeights = classWeights'/mean(classWeights);
numClasses  = numel(classes);

timePoolSize = ceil(numHops/8);

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer([numHops,afe.FeatureVectorLength])
    
    convolution2dLayer(3,numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,2*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([timePoolSize,1])
    dropoutLayer(dropoutProb)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer('Classes', classes, 'ClassWeights', classWeights) ];
%% 
miniBatchSize = 16;

validationFrequency = floor(numel(YTrain)/miniBatchSize)

options = trainingOptions("adam", ...
    InitialLearnRate=3e-4, ...
    MaxEpochs=15, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=true);
%% 

trainedNet = trainNetwork(XTrain,YTrain,layers,options);

%% 

YTrainPred = classify(trainedNet, XTrain);
trainError  = mean(YTrainPred ~= YTrain);
disp("Training Error:" + trainError*100 + "%")


%% 

figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
cm = confusionchart(YTrain,YTrainPred, ...
    Title="Confusion Matrix for Train Data", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
sortClasses(cm,[commands])




