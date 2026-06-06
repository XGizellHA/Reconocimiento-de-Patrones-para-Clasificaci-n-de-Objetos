function distorsionTotal = vectorQuantization(pixeles, codebook)
% Calcula la distorsión total al cuantizar un conjunto de píxeles con un codebook

pixeles = double(pixeles);
N = size(pixeles, 1);
numCentroides = size(codebook, 1);

% Para cada píxel, encontrar el centroide más cercano
distancias = zeros(N, numCentroides);
for i = 1:numCentroides
    centroide = codebook(i, :);
    diff = pixeles - centroide;
    distancias(:, i) = sum(diff.^2, 2);
end

[minDistancias, ~] = min(distancias, [], 2);
distorsionTotal = mean(minDistancias);
end