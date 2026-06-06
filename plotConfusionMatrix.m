function [imagenesEntrenamiento, imagenesPrueba, etiquetasEntrenamiento, etiquetasPrueba] = ...
    loadImages(clases, numEntrenamiento, numPrueba)
% Carga las imágenes de las carpetas y las divide en entrenamiento y prueba

numClases = length(clases);
imagenesEntrenamiento = cell(numClases, 1);
imagenesPrueba = cell(numClases, 1);
etiquetasEntrenamiento = [];
etiquetasPrueba = [];

for i = 1:numClases
    % Ruta de la carpeta
    folderPath = fullfile('Objetos_segmentados', clases{i});

    % Obtener lista de archivos
    archivos = dir(fullfile(folderPath, '*.jpg'));
    if isempty(archivos)
        archivos = dir(fullfile(folderPath, '*.png'));
    end
    if isempty(archivos)
        archivos = dir(fullfile(folderPath, '*.jpeg'));
    end

    % Verificar que hay suficientes imágenes
    if length(archivos) < numEntrenamiento + numPrueba
        warning('Clase %s solo tiene %d imágenes', clases{i}, length(archivos));
    end

    % Cargar imágenes de entrenamiento
    imagenesEntrenamiento{i} = cell(numEntrenamiento, 1);
    for j = 1:numEntrenamiento
        imgPath = fullfile(folderPath, archivos(j).name);
        imagenesEntrenamiento{i}{j} = imread(imgPath);
        etiquetasEntrenamiento = [etiquetasEntrenamiento; i];
    end

    % Cargar imágenes de prueba
    imagenesPrueba{i} = cell(numPrueba, 1);
    for j = 1:numPrueba
        idx = numEntrenamiento + j;
        if idx <= length(archivos)
            imgPath = fullfile(folderPath, archivos(idx).name);
            imagenesPrueba{i}{j} = imread(imgPath);
            etiquetasPrueba = [etiquetasPrueba; i];
        end
    end
end

fprintf('  Clases cargadas: %d\n', numClases);
for i = 1:numClases
    fprintf('    %s: %d entrenamiento, %d prueba\n', clases{i}, ...
        length(imagenesEntrenamiento{i}), length(imagenesPrueba{i}));
end
end