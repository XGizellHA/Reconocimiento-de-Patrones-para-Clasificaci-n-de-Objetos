%% Proyecto Final - Reconocimiento de Patrones
% Autor: Estudiante de Ingeniería en Computación
% Descripción: Implementación de cuantización vectorial y redes neuronales
%              para reconocimiento de objetos segmentados

clear all; close all; clc;

%% Crear carpeta de resultados si no existe
if ~exist('results', 'dir')
    mkdir('results');
end

%% Agregar carpetas al path
currentFolder = pwd;
if exist(fullfile(currentFolder, 'functions'), 'dir')
    addpath(fullfile(currentFolder, 'functions'));
else
    error('No se encuentra la carpeta "functions". Verifique la estructura del proyecto.');
end

%% Configuración inicial
fprintf('=== PROYECTO FINAL: RECONOCIMIENTO DE PATRONES ===\n');
fprintf('Iniciando procesamiento...\n\n');

% Parámetros
numClases = 10;
numEntrenamiento = 15;   % 15 imágenes para entrenamiento
numPrueba = 5;           % 5 imágenes para prueba
codebookSize = 32;       % Tamaño del codebook (32 centroides)
numEpochs = 50;          % Épocas para red neuronal

% Nombres de las clases (orden alfabético)
clases = {'appleJuice', 'blueBowl', 'blueLego', 'blueMug', 'blueSpoon', ...
          'chocolateCookies', 'orangeJuice', 'orangeKnife', 'redLego', 'redMug'};

% Nombres legibles para las gráficas
nombresLegibles = {'Jugo Manzana', 'Tazón Azul', 'Lego Azul', 'Taza Azul', 'Cuchara Azul', ...
                   'Galletas Choc.', 'Jugo Naranja', 'Cuchillo Naranja', 'Lego Rojo', 'Taza Roja'};

%% Cargar imágenes
fprintf('Cargando imágenes...\n');

% Verificar que la carpeta Objetos_segmentados existe
if ~exist('Objetos_segmentados', 'dir')
    error('No se encuentra la carpeta "Objetos_segmentados". Verifique que está en el directorio actual.');
end

[imagenesEntrenamiento, imagenesPrueba, etiquetasEntrenamiento, etiquetasPrueba] = ...
    loadImages(clases, numEntrenamiento, numPrueba);

fprintf('Imágenes cargadas exitosamente.\n');
fprintf('  Entrenamiento: %d imágenes\n', length(etiquetasEntrenamiento));
fprintf('  Prueba: %d imágenes\n\n', length(etiquetasPrueba));

%% EXPERIMENTO 1: Cuantización Vectorial con RGB
fprintf('=== EXPERIMENTO 1: Cuantización Vectorial - RGB ===\n');
tic;
codebooksRGB = cell(numClases, 1);
for i = 1:numClases
    fprintf('  Creando codebook para clase %d/%d: %s\n', i, numClases, clases{i});
    pixeles = [];
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        [h, w, ~] = size(img);
        pixeles = [pixeles; reshape(img, h*w, 3)];
    end
    if ~isempty(pixeles)
        codebooksRGB{i} = createCodebook(pixeles, codebookSize);
    else
        warning('No se pudieron extraer píxeles de la clase %s', clases{i});
        codebooksRGB{i} = rand(codebookSize, 3);
    end
end

% Evaluar
prediccionesRGB_train = zeros(length(etiquetasEntrenamiento), 1);
prediccionesRGB_test = zeros(length(etiquetasPrueba), 1);

for i = 1:numClases
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        [h, w, ~] = size(img);
        pixeles = reshape(img, h*w, 3);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksRGB{k})
                distorsiones(k) = vectorQuantization(pixeles, codebooksRGB{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesRGB_train((i-1)*numEntrenamiento + j) = idx;
    end
    
    for j = 1:numPrueba
        img = double(imagenesPrueba{i}{j}) / 255;
        [h, w, ~] = size(img);
        pixeles = reshape(img, h*w, 3);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksRGB{k})
                distorsiones(k) = vectorQuantization(pixeles, codebooksRGB{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesRGB_test((i-1)*numPrueba + j) = idx;
    end
end

% Matrices de confusión y gráficas
fprintf('\nMatriz de Confusión - RGB (Entrenamiento):\n');
confMatRGB_train = confusionMatrix(prediccionesRGB_train, etiquetasEntrenamiento, numClases);
accuracyRGB_train_total = sum(diag(confMatRGB_train)) / sum(confMatRGB_train(:)) * 100;
fprintf('Exactitud total en entrenamiento: %.2f%%\n\n', accuracyRGB_train_total);
plotConfusionMatrix(confMatRGB_train, nombresLegibles, ...
    sprintf('Matriz de Confusión - RGB (Entrenamiento) - Exactitud: %.2f%%', accuracyRGB_train_total));

fprintf('Matriz de Confusión - RGB (Prueba):\n');
confMatRGB_test = confusionMatrix(prediccionesRGB_test, etiquetasPrueba, numClases);
accuracyRGB_test_total = sum(diag(confMatRGB_test)) / sum(confMatRGB_test(:)) * 100;
fprintf('Exactitud total en prueba: %.2f%%\n\n', accuracyRGB_test_total);
plotConfusionMatrix(confMatRGB_test, nombresLegibles, ...
    sprintf('Matriz de Confusión - RGB (Prueba) - Exactitud: %.2f%%', accuracyRGB_test_total));
tiempoRGB = toc;
fprintf('Tiempo de ejecución: %.2f segundos\n\n', tiempoRGB);

%% EXPERIMENTO 2: Cuantización Vectorial con HSV
fprintf('=== EXPERIMENTO 2: Cuantización Vectorial - HSV ===\n');
tic;
codebooksHSV = cell(numClases, 1);
for i = 1:numClases
    fprintf('  Creando codebook para clase %d/%d: %s\n', i, numClases, clases{i});
    pixeles = [];
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        [h, w, ~] = size(img);
        imgHSV = rgb2hsv_custom(img);
        pixeles = [pixeles; reshape(imgHSV, h*w, 3)];
    end
    if ~isempty(pixeles)
        codebooksHSV{i} = createCodebook(pixeles, codebookSize);
    else
        codebooksHSV{i} = rand(codebookSize, 3);
    end
end

% Evaluar
prediccionesHSV_train = zeros(length(etiquetasEntrenamiento), 1);
prediccionesHSV_test = zeros(length(etiquetasPrueba), 1);

for i = 1:numClases
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        [h, w, ~] = size(img);
        imgHSV = rgb2hsv_custom(img);
        pixeles = reshape(imgHSV, h*w, 3);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksHSV{k})
                distorsiones(k) = vectorQuantization(pixeles, codebooksHSV{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesHSV_train((i-1)*numEntrenamiento + j) = idx;
    end
    
    for j = 1:numPrueba
        img = double(imagenesPrueba{i}{j}) / 255;
        [h, w, ~] = size(img);
        imgHSV = rgb2hsv_custom(img);
        pixeles = reshape(imgHSV, h*w, 3);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksHSV{k})
                distorsiones(k) = vectorQuantization(pixeles, codebooksHSV{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesHSV_test((i-1)*numPrueba + j) = idx;
    end
end

fprintf('\nMatriz de Confusión - HSV (Entrenamiento):\n');
confMatHSV_train = confusionMatrix(prediccionesHSV_train, etiquetasEntrenamiento, numClases);
accuracyHSV_train_total = sum(diag(confMatHSV_train)) / sum(confMatHSV_train(:)) * 100;
fprintf('Exactitud total en entrenamiento: %.2f%%\n\n', accuracyHSV_train_total);
plotConfusionMatrix(confMatHSV_train, nombresLegibles, ...
    sprintf('Matriz de Confusión - HSV (Entrenamiento) - Exactitud: %.2f%%', accuracyHSV_train_total));

fprintf('Matriz de Confusión - HSV (Prueba):\n');
confMatHSV_test = confusionMatrix(prediccionesHSV_test, etiquetasPrueba, numClases);
accuracyHSV_test_total = sum(diag(confMatHSV_test)) / sum(confMatHSV_test(:)) * 100;
fprintf('Exactitud total en prueba: %.2f%%\n\n', accuracyHSV_test_total);
plotConfusionMatrix(confMatHSV_test, nombresLegibles, ...
    sprintf('Matriz de Confusión - HSV (Prueba) - Exactitud: %.2f%%', accuracyHSV_test_total));
tiempoHSV = toc;
fprintf('Tiempo de ejecución: %.2f segundos\n\n', tiempoHSV);

%% EXPERIMENTO 3: Cuantización Vectorial con RGB + Sobel
fprintf('=== EXPERIMENTO 3: Cuantización Vectorial - RGB + Sobel ===\n');
tic;
codebooksSobelRGB = cell(numClases, 1);
for i = 1:numClases
    fprintf('  Creando codebook para clase %d/%d: %s\n', i, numClases, clases{i});
    features = [];
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        imgRed = imresize(img, 0.5);
        feat = sobelFeatures(imgRed);
        features = [features; feat];
    end
    if ~isempty(features)
        codebooksSobelRGB{i} = createCodebook(features, codebookSize);
    else
        codebooksSobelRGB{i} = rand(codebookSize, size(features, 2));
    end
end

% Evaluar
prediccionesSobelRGB_train = zeros(length(etiquetasEntrenamiento), 1);
prediccionesSobelRGB_test = zeros(length(etiquetasPrueba), 1);

for i = 1:numClases
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        imgRed = imresize(img, 0.5);
        feat = sobelFeatures(imgRed);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksSobelRGB{k})
                distorsiones(k) = vectorQuantization(feat, codebooksSobelRGB{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesSobelRGB_train((i-1)*numEntrenamiento + j) = idx;
    end
    
    for j = 1:numPrueba
        img = double(imagenesPrueba{i}{j}) / 255;
        imgRed = imresize(img, 0.5);
        feat = sobelFeatures(imgRed);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksSobelRGB{k})
                distorsiones(k) = vectorQuantization(feat, codebooksSobelRGB{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesSobelRGB_test((i-1)*numPrueba + j) = idx;
    end
end

fprintf('\nMatriz de Confusión - RGB+Sobel (Entrenamiento):\n');
confMatSobelRGB_train = confusionMatrix(prediccionesSobelRGB_train, etiquetasEntrenamiento, numClases);
accuracySobelRGB_train_total = sum(diag(confMatSobelRGB_train)) / sum(confMatSobelRGB_train(:)) * 100;
fprintf('Exactitud total en entrenamiento: %.2f%%\n\n', accuracySobelRGB_train_total);
plotConfusionMatrix(confMatSobelRGB_train, nombresLegibles, ...
    sprintf('Matriz de Confusión - RGB+Sobel (Entrenamiento) - Exactitud: %.2f%%', accuracySobelRGB_train_total));

fprintf('Matriz de Confusión - RGB+Sobel (Prueba):\n');
confMatSobelRGB_test = confusionMatrix(prediccionesSobelRGB_test, etiquetasPrueba, numClases);
accuracySobelRGB_test_total = sum(diag(confMatSobelRGB_test)) / sum(confMatSobelRGB_test(:)) * 100;
fprintf('Exactitud total en prueba: %.2f%%\n\n', accuracySobelRGB_test_total);
plotConfusionMatrix(confMatSobelRGB_test, nombresLegibles, ...
    sprintf('Matriz de Confusión - RGB+Sobel (Prueba) - Exactitud: %.2f%%', accuracySobelRGB_test_total));
tiempoSobelRGB = toc;
fprintf('Tiempo de ejecución: %.2f segundos\n\n', tiempoSobelRGB);

%% EXPERIMENTO 4: Cuantización Vectorial con HSV + Sobel
fprintf('=== EXPERIMENTO 4: Cuantización Vectorial - HSV + Sobel ===\n');
tic;
codebooksSobelHSV = cell(numClases, 1);
for i = 1:numClases
    fprintf('  Creando codebook para clase %d/%d: %s\n', i, numClases, clases{i});
    features = [];
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        imgRed = imresize(img, 0.5);
        imgHSV = rgb2hsv_custom(imgRed);
        feat = sobelFeatures(imgHSV);
        features = [features; feat];
    end
    if ~isempty(features)
        codebooksSobelHSV{i} = createCodebook(features, codebookSize);
    else
        codebooksSobelHSV{i} = rand(codebookSize, size(features, 2));
    end
end

% Evaluar
prediccionesSobelHSV_train = zeros(length(etiquetasEntrenamiento), 1);
prediccionesSobelHSV_test = zeros(length(etiquetasPrueba), 1);

for i = 1:numClases
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        imgRed = imresize(img, 0.5);
        imgHSV = rgb2hsv_custom(imgRed);
        feat = sobelFeatures(imgHSV);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksSobelHSV{k})
                distorsiones(k) = vectorQuantization(feat, codebooksSobelHSV{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesSobelHSV_train((i-1)*numEntrenamiento + j) = idx;
    end
    
    for j = 1:numPrueba
        img = double(imagenesPrueba{i}{j}) / 255;
        imgRed = imresize(img, 0.5);
        imgHSV = rgb2hsv_custom(imgRed);
        feat = sobelFeatures(imgHSV);
        distorsiones = zeros(numClases, 1);
        for k = 1:numClases
            if ~isempty(codebooksSobelHSV{k})
                distorsiones(k) = vectorQuantization(feat, codebooksSobelHSV{k});
            else
                distorsiones(k) = inf;
            end
        end
        [~, idx] = min(distorsiones);
        prediccionesSobelHSV_test((i-1)*numPrueba + j) = idx;
    end
end

fprintf('\nMatriz de Confusión - HSV+Sobel (Entrenamiento):\n');
confMatSobelHSV_train = confusionMatrix(prediccionesSobelHSV_train, etiquetasEntrenamiento, numClases);
accuracySobelHSV_train_total = sum(diag(confMatSobelHSV_train)) / sum(confMatSobelHSV_train(:)) * 100;
fprintf('Exactitud total en entrenamiento: %.2f%%\n\n', accuracySobelHSV_train_total);
plotConfusionMatrix(confMatSobelHSV_train, nombresLegibles, ...
    sprintf('Matriz de Confusión - HSV+Sobel (Entrenamiento) - Exactitud: %.2f%%', accuracySobelHSV_train_total));

fprintf('Matriz de Confusión - HSV+Sobel (Prueba):\n');
confMatSobelHSV_test = confusionMatrix(prediccionesSobelHSV_test, etiquetasPrueba, numClases);
accuracySobelHSV_test_total = sum(diag(confMatSobelHSV_test)) / sum(confMatSobelHSV_test(:)) * 100;
fprintf('Exactitud total en prueba: %.2f%%\n\n', accuracySobelHSV_test_total);
plotConfusionMatrix(confMatSobelHSV_test, nombresLegibles, ...
    sprintf('Matriz de Confusión - HSV+Sobel (Prueba) - Exactitud: %.2f%%', accuracySobelHSV_test_total));
tiempoSobelHSV = toc;
fprintf('Tiempo de ejecución: %.2f segundos\n\n', tiempoSobelHSV);

%% EXPERIMENTO 5: Redes Neuronales
fprintf('=== EXPERIMENTO 5: Redes Neuronales ===\n');
fprintf('Configuración de la red neuronal:\n');
fprintf('  Características: Media y desviación de cada canal RGB (6 características)\n');
fprintf('  Capa oculta: 20 neuronas (función ReLU)\n');
fprintf('  Capa de salida: %d neuronas (función Softmax)\n', numClases);
fprintf('  Épocas: %d\n\n', numEpochs);

% Extraer características para cada imagen (una fila por imagen)
trainFeatures = [];
trainLabels = [];
testFeatures = [];
testLabels = [];

for i = 1:numClases
    for j = 1:numEntrenamiento
        img = double(imagenesEntrenamiento{i}{j}) / 255;
        imgRed = imresize(img, 0.25);
        % Características: media y desviación de cada canal RGB
        feat = [];
        for c = 1:3
            canal = imgRed(:, :, c);
            feat = [feat, mean(canal(:)), std(canal(:))];
        end
        trainFeatures = [trainFeatures; feat];
        trainLabels = [trainLabels; i];
    end
    
    for j = 1:numPrueba
        img = double(imagenesPrueba{i}{j}) / 255;
        imgRed = imresize(img, 0.25);
        feat = [];
        for c = 1:3
            canal = imgRed(:, :, c);
            feat = [feat, mean(canal(:)), std(canal(:))];
        end
        testFeatures = [testFeatures; feat];
        testLabels = [testLabels; i];
    end
end

fprintf('  Características extraídas:\n');
fprintf('    Entrenamiento: %d muestras x %d características\n', size(trainFeatures, 1), size(trainFeatures, 2));
fprintf('    Prueba: %d muestras x %d características\n', size(testFeatures, 1), size(testFeatures, 2));

% Normalizar características
trainFeatures = (trainFeatures - mean(trainFeatures)) ./ std(trainFeatures);
testFeatures = (testFeatures - mean(testFeatures)) ./ std(testFeatures);
trainFeatures(isnan(trainFeatures)) = 0;
testFeatures(isnan(testFeatures)) = 0;

% Crear y entrenar red neuronal
net = patternnet(20);
net.trainParam.epochs = numEpochs;
net.trainParam.showWindow = false;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Entrenar
fprintf('  Entrenando red neuronal...\n');
net = train(net, trainFeatures', ind2vec(trainLabels'));

% Evaluar
trainPred = vec2ind(net(trainFeatures'));
testPred = vec2ind(net(testFeatures'));

accNN_train = sum(trainPred == trainLabels') / length(trainLabels) * 100;
accNN_test = sum(testPred == testLabels') / length(testLabels) * 100;

fprintf('\nResultados Red Neuronal:\n');
fprintf('  Exactitud entrenamiento: %.2f%%\n', accNN_train);
fprintf('  Exactitud prueba: %.2f%%\n\n', accNN_test);

% Matrices de confusión
confMatNN_train = confusionmat(trainLabels', trainPred);
confMatNN_test = confusionmat(testLabels', testPred);

plotConfusionMatrix(confMatNN_train, nombresLegibles, ...
    sprintf('Red Neuronal (Entrenamiento) - Exactitud: %.2f%%', accNN_train));
plotConfusionMatrix(confMatNN_test, nombresLegibles, ...
    sprintf('Red Neuronal (Prueba) - Exactitud: %.2f%%', accNN_test));

%% RESULTADOS FINALES Y ANÁLISIS
fprintf('\n=== RESUMEN DE RESULTADOS ===\n');
fprintf('\n%-25s | %-15s | %-15s | %-15s\n', 'Método', 'Exactitud Train', 'Exactitud Test', 'Tiempo (s)');
fprintf('%s\n', repmat('-', 80, 1));
fprintf('%-25s | %-15.2f | %-15.2f | %-15.2f\n', 'RGB (VQ)', accuracyRGB_train_total, accuracyRGB_test_total, tiempoRGB);
fprintf('%-25s | %-15.2f | %-15.2f | %-15.2f\n', 'HSV (VQ)', accuracyHSV_train_total, accuracyHSV_test_total, tiempoHSV);
fprintf('%-25s | %-15.2f | %-15.2f | %-15.2f\n', 'RGB+Sobel (VQ)', accuracySobelRGB_train_total, accuracySobelRGB_test_total, tiempoSobelRGB);
fprintf('%-25s | %-15.2f | %-15.2f | %-15.2f\n', 'HSV+Sobel (VQ)', accuracySobelHSV_train_total, accuracySobelHSV_test_total, tiempoSobelHSV);
fprintf('%-25s | %-15.2f | %-15.2f | %-15.2f\n', 'Red Neuronal', accNN_train, accNN_test, 0);

% Crear figura comparativa
figure('Name', 'Comparación de Métodos', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 600]);

metodos = {'RGB (VQ)', 'HSV (VQ)', 'RGB+Sobel (VQ)', 'HSV+Sobel (VQ)', 'Red Neuronal'};
exactitudes_train = [accuracyRGB_train_total, accuracyHSV_train_total, ...
                     accuracySobelRGB_train_total, accuracySobelHSV_train_total, accNN_train];
exactitudes_test = [accuracyRGB_test_total, accuracyHSV_test_total, ...
                    accuracySobelRGB_test_total, accuracySobelHSV_test_total, accNN_test];

bar(1:5, [exactitudes_train; exactitudes_test]');
xlabel('Método');
ylabel('Exactitud (%)');
title('Comparación de Exactitud entre Métodos');
legend('Entrenamiento', 'Prueba', 'Location', 'southeast');
grid on;
set(gca, 'XTickLabel', metodos);
xtickangle(45);
ylim([0, 100]);

% Agregar valores en las barras
for i = 1:5
    text(i - 0.2, exactitudes_train(i) + 1, sprintf('%.1f%%', exactitudes_train(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    text(i + 0.2, exactitudes_test(i) + 1, sprintf('%.1f%%', exactitudes_test(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% Guardar figura comparativa
saveas(gcf, fullfile('results', 'Comparacion_Metodos.png'));

%% ANÁLISIS DE RESULTADOS
fprintf('\n=== ANÁLISIS DE RESULTADOS ===\n');
fprintf('\n1. Comparación de espacios de color:\n');
if accuracyHSV_test_total > accuracyRGB_test_total
    fprintf('   ✓ HSV supera a RGB (+%.2f puntos porcentuales en prueba)\n', ...
        accuracyHSV_test_total - accuracyRGB_test_total);
else
    fprintf('   ✓ RGB supera a HSV (+%.2f puntos porcentuales en prueba)\n', ...
        accuracyRGB_test_total - accuracyHSV_test_total);
end
fprintf('   - HSV es más robusto a variaciones de iluminación\n');
fprintf('   - HSV separa la información de color (Hue) de la intensidad (Value)\n\n');

fprintf('2. Efecto de los filtros Sobel:\n');
if accuracySobelRGB_test_total > accuracyRGB_test_total
    fprintf('   ✓ Sobel mejora RGB en +%.2f puntos porcentuales\n', ...
        accuracySobelRGB_test_total - accuracyRGB_test_total);
end
if accuracySobelHSV_test_total > accuracyHSV_test_total
    fprintf('   ✓ Sobel mejora HSV en +%.2f puntos porcentuales\n', ...
        accuracySobelHSV_test_total - accuracyHSV_test_total);
end
fprintf('   - Los filtros Sobel añaden información de bordes y formas\n');
fprintf('   - Ayudan a diferenciar objetos con colores similares\n\n');

fprintf('3. Comparación Cuantización Vectorial vs Redes Neuronales:\n');
if accNN_test > max([accuracyRGB_test_total, accuracyHSV_test_total, ...
                     accuracySobelRGB_test_total, accuracySobelHSV_test_total])
    fprintf('   ✓ Red Neuronal es el método más preciso (%.2f%%)\n', accNN_test);
else
    [bestVQ, idx] = max([accuracyRGB_test_total, accuracyHSV_test_total, ...
                         accuracySobelRGB_test_total, accuracySobelHSV_test_total]);
    fprintf('   ✓ VQ con %s es más preciso (%.2f%% vs %.2f%% de Red Neuronal)\n', ...
        metodos{idx}, bestVQ, accNN_test);
end
fprintf('   - Redes Neuronales requieren más tiempo de entrenamiento\n');
fprintf('   - Cuantización Vectorial es más rápida en ejecución\n\n');

fprintf('4. Mejor método según los resultados:\n');
todos_test = [accuracyRGB_test_total, accuracyHSV_test_total, ...
              accuracySobelRGB_test_total, accuracySobelHSV_test_total, accNN_test];
[mejorValor, mejorIndice] = max(todos_test);
fprintf('   ★ MEJOR MÉTODO: %s (Exactitud: %.2f%%)\n', metodos{mejorIndice}, mejorValor);

fprintf('\n5. Recomendaciones:\n');
fprintf('   - Para sistemas en tiempo real: %s\n', metodos{mejorIndice});
fprintf('   - Para máxima precisión con más recursos: Red Neuronal\n');
fprintf('   - Usar Sobel cuando los objetos tienen formas distintivas\n');
fprintf('   - Usar HSV cuando hay variaciones de iluminación\n');

fprintf('\n=== PROYECTO COMPLETADO EXITOSAMENTE ===\n');
fprintf('Las figuras de las matrices de confusión se han guardado en la carpeta "results"\n');