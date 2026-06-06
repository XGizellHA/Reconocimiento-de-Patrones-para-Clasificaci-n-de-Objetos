function codebook = createCodebook(pixeles, codebookSize)
% Implementa el algoritmo de Linde-Buzo-Gray (LBG) para crear un codebook
% de cuantización vectorial

pixeles = double(pixeles);
N = size(pixeles, 1);
dim = size(pixeles, 2);

% Inicializar con el centroide de todos los vectores
codebook = mean(pixeles, 1)';
codebook = codebook(:);

% Aplicar algoritmo LBG para obtener el número deseado de centroides
numCentroides = 1;
while numCentroides < codebookSize
    % Duplicar centroides con pequeña perturbación
    epsilon = 0.01;
    codebook_new = [];
    for i = 1:size(codebook, 2)
        c = codebook(:, i);
        codebook_new = [codebook_new, c * (1 + epsilon), c * (1 - epsilon)];
    end
    codebook = codebook_new;
    numCentroides = size(codebook, 2);

    % Iteraciones de Lloyd-Max
    maxIter = 50;
    for iter = 1:maxIter
        % Asignar cada punto al centroide más cercano
        distancias = zeros(N, numCentroides);
        for i = 1:numCentroides
            centroide = codebook(:, i);
            centroide = centroide(:)';
            diff = pixeles - centroide;
            distancias(:, i) = sum(diff.^2, 2);
        end

        [~, asignaciones] = min(distancias, [], 2);

        % Actualizar centroides
        codebook_new = zeros(dim, numCentroides);
        for i = 1:numCentroides
            indices = (asignaciones == i);
            if sum(indices) > 0
                codebook_new(:, i) = mean(pixeles(indices, :), 1)';
            else
                codebook_new(:, i) = codebook(:, i);
            end
        end

        % Verificar convergencia
        if norm(codebook_new - codebook, 'fro') < 1e-6
            codebook = codebook_new;
            break;
        end
        codebook = codebook_new;
    end
end

% Asegurar el tamaño correcto
codebook = codebook';
end