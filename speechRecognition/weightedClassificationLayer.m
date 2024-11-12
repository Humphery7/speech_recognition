classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
    % A custom layer to compute weighted cross-entropy loss
    properties
        classWeights
    end
    
    methods
        % Constructor
        function layer = weightedClassificationLayer(classWeights, name)
            % Set default layer name if not provided
            if nargin < 2
                name = 'weightedClassificationLayer';
            end
            
            % Call the superclass constructor
            layer.Name = name;
            layer.classWeights = classWeights;
        end
        
        % Forward pass (compute loss)
        function loss = forwardLoss(layer, Y, T)
            % Y is the predicted logits or probabilities (network output)
            % T is the true labels (ground truth)
            
            % Compute the cross-entropy loss
            ceLoss = crossentropy(Y, T);
            
            % Apply class weights (element-wise)
            weightedLoss = sum(layer.classWeights .* ceLoss);
            
            % Return the final loss as a scalar
            loss = weightedLoss;
        end
        
        % Backward pass (compute gradients)
        function gradients = backwardLoss(layer, Y, T)
            % Compute the gradient of the weighted loss
            
            % Compute the loss gradient (dL/dY) using the cross-entropy derivative
            dL_dY = (Y - T);
            
            % Apply the class weights to the gradient
            gradients = layer.classWeights .* dL_dY;
        end
    end
end
