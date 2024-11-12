% Ensure you have the correct trained network loaded
% Example: trainedNet = load('your_trained_network.mat'); 
% For now, let's assume `trainedNet` is the trained network variable

% 1. Capture live audio
fs = 48000;  % Sampling rate for audio (same as training)
segmentDuration = 3;  % Duration for segment (in seconds)
frameDuration = 0.025;  % Frame duration (in seconds)
hopDuration = 0.01;  % Hop duration (in seconds)

% Setup audio recording object
recObj = audiorecorder(fs, 16, 1);  % 16-bit depth, mono channel
disp('Recording audio...');
recordblocking(recObj, segmentDuration);  % Record for 'segmentDuration' seconds
disp('Recording complete');

% Convert recorded audio to an array
audioData = getaudiodata(recObj);  % audioData is now a column vector
% Play the recorded audio back
disp('Playing back recorded audio...');
sound(audioData, fs);  % Play the audio using the same sample rate


% 2. Preprocess the audio (same steps as training)
segmentSamples = round(segmentDuration * fs);
frameSamples = round(frameDuration * fs);
hopSamples = round(hopDuration * fs);
overlapSamples = frameSamples - hopSamples;

% Feature extraction setup (matching training parameters)
afe = audioFeatureExtractor(SampleRate=fs, ...
    FFTLength=2048, ...
    window=hann(frameSamples, "periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);

setExtractorParameters(afe, "barkSpectrum", NumBands=50, WindowNormalization=false);

% Padding audio to match segment duration
numSamples = length(audioData);
numToPadFront = floor((segmentSamples - numSamples) / 2);
numToPadBack = ceil((segmentSamples - numSamples) / 2);
audioPadded = [zeros(numToPadFront, 1); audioData; zeros(numToPadBack, 1)];

% Extract features (Bark Spectrum)
features = extract(afe, audioPadded);

% 3. Prepare the features for input into the trained network
% The feature matrix needs to be reshaped and formatted to match the input layer of the network
% Assuming trained network expects data of size: [numHops, numBands, 1] or [numHops, numBands, 2] (if stereo)

% Reshape the features
XLive = reshape(features, [size(features, 1), size(features, 2), 1, 1]);  % Reshaping to fit input size (with 1 channel)

% 4. Classify the live audio
% Perform prediction using the trained network
[YPred, scores] = classify(trainedNet, XLive);

% 5. Display the predicted class
disp(['Predicted Command: ', char(YPred)]);  % Display the predicted command
disp('Prediction Scores:');
disp(scores);  % Display the score for each class

% Optionally, you can also visualize the audio waveform and spectrogram
figure;

% Plot the waveform
subplot(2, 1, 1);
t = (0:length(audioPadded)-1) / fs;
plot(t, audioPadded);
title('Audio Waveform');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the Bark Spectrum
subplot(2, 1, 2);
imagesc(10*log10(features.'));  % Convert to dB scale for visualization
axis xy;
colormap jet;
colorbar;
title('Bark Spectrum');
xlabel('Time Frames');
ylabel('Frequency Bands');
