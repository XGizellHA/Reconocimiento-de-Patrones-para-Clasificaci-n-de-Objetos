function [accuracyTrain, accuracyTest, confMatTrain, confMatTest] = neuralNetworkClassifier(...
    imagenesEntrenamiento, imagenesPrueba, etiquetasEntrenamiento, ...
    etiquetasPrueba, numClases, numEntrenamiento, numPrueba, tipo, numEpochs)
% Implementa un clasificador basado en red neuronal feedforward

% Extraer características de todas las imágenes de entrenamiento
trainFeatures = [];
trainLabels = [];
testFeatures = [];
testLabels = [];

for i = 1:numClases
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        imgRed = imresize(img, 0.25);
        
        switch tipo
            case 'rgb'
                feat = extractRGBFeaturesReduced(imgRed);
            case 'hsv'
                imgHSV = rgb2hsv_custom(imgRed);
                feat = extractRGBFeaturesReduced(imgHSV);
            case 'rgb_sobel'
                feat = extractSobelFeaturesReduced(imgRed);
            case 'hsv_sobel'
                imgHSV = rgb2hsv_custom(imgRed);
                feat = extractSobelFeaturesReduced(imgHSV);
            otherwise
                feat = extractRGBFeaturesReduced(imgRed);
        end
        
        % Una sola fila por imagen
        trainFeatures = [trainFeatures; feat];
        trainLabels = [trainLabels; i];
    end
    
    for j = 1:numPrueba
        img = double(imagenesPrueba{i}{j}) / 255;
        imgRed = imresize(img, 0.25);
        
        switch tipo
            case 'rgb'
                feat = extractRGBFeaturesReduced(imgRed);
            case 'hsv'
                imgHSV = rgb2hsv_custom(imgRed);
                feat = extractRGBFeaturesReduced(imgHSV);
            case 'rgb_sobel'
                feat = extractSobelFeaturesReduced(imgRed);
            case 'hsv_sobel'
                imgHSV = rgb2hsv_custom(imgRed);
                feat = extractSobelFeaturesReduced(imgHSV);
            otherwise
                feat = extractRGBFeaturesReduced(imgRed);
        end
        
        testFeatures = [testFeatures; feat];
        testLabels = [testLabels; i];
    end
end

% Verificar dimensiones
fprintf('    Dimensiones de características de entrenamiento: %d x %d\n', size(trainFeatures));
fprintf('    Número de etiquetas de entrenamiento: %d\n', length(trainLabels));
fprintf('    Dimensiones de características de prueba: %d x %d\n', size(testFeatures));
fprintf('    Número de etiquetas de prueba: %d\n', length(testLabels));

% Normalizar características
trainFeatures = (trainFeatures - mean(trainFeatures)) ./ std(trainFeatures);
testFeatures = (testFeatures - mean(testFeatures)) ./ std(testFeatures);

% Manejar valores NaN (si hay varianza cero)
trainFeatures(isnan(trainFeatures)) = 0;
testFeatures(isnan(testFeatures)) = 0;

% Crear y entrenar la red neuronal
net = patternnet(20);  % 20 neuronas en capa oculta
net.trainParam.epochs = numEpochs;
net.trainParam.showWindow = false;  % No mostrar ventana de entrenamiento
net.divideParam.trainRatio = 1.0;   % Usar todas las muestras para entrenamiento
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

% Entrenar
net = train(net, trainFeatures', ind2vec(trainLabels'));

% Evaluar
trainPred = vec2ind(net(trainFeatures'));
testPred = vec2ind(net(testFeatures'));

% Calcular exactitudes
accuracyTrain = sum(trainPred == trainLabels') / length(trainLabels) * 100;
accuracyTest = sum(testPred == testLabels') / length(testLabels) * 100;

% Calcular matrices de confusión
confMatTrain = confusionmat(trainLabels', trainPred);
confMatTest = confusionmat(testLabels', testPred);
end

function feat = extractRGBFeaturesReduced(img)
% Extrae características estadísticas de una imagen RGB
% Retorna un vector de 6 elementos: media y desviación de cada canal
[h, w, ~] = size(img);
feat = zeros(1, 6);
idx = 1;
for c = 1:3
    canal = img(:, :, c);
    feat(idx) = mean(canal(:));
    feat(idx+1) = std(canal(:));
    idx = idx + 2;
end
end

function feat = extractSobelFeaturesReduced(img)
% Extrae características estadísticas de bordes Sobel
% Retorna un vector de 12 elementos: estadísticas de gradientes
[h, w, ~] = size(img);

% Filtros Sobel
sobelH = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
sobelV = [-1, -2, -1; 0, 0, 0; 1, 2, 1];

feat = zeros(1, 12);
idx = 1;

for c = 1:3
    canal = img(:, :, c);
    
    % Gradiente horizontal
    gradH = conv2(canal, sobelH, 'same');
    % Gradiente vertical
    gradV = conv2(canal, sobelV, 'same');
    % Magnitud del gradiente
    gradMag = sqrt(gradH.^2 + gradV.^2);
    
    % Estadísticas
    feat(idx) = mean(gradH(:));
    feat(idx+1) = std(gradH(:));
    feat(idx+2) = mean(gradV(:));
    feat(idx+3) = std(gradV(:));
    idx = idx + 4;
end
end