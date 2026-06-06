function features = sobelFeatures(img)
% Extrae características utilizando filtros Sobel y concatena con valores RGB/HSV
% Basado en la teoría de la lección 5

% Filtros Sobel
sobelH = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
sobelV = [-1, -2, -1; 0, 0, 0; 1, 2, 1];

[h, w, dim] = size(img);
features = zeros(h * w, dim + 2);

% Aplicar filtros Sobel a cada canal
for d = 1:dim
    canal = img(:, :, d);

    % Gradiente horizontal
    gradH = conv2(canal, sobelH, 'same');
    % Gradiente vertical
    gradV = conv2(canal, sobelV, 'same');

    % Concatenar características
    idx = 1;
    for i = 1:h
        for j = 1:w
            features(idx, :) = [reshape(img(i, j, :), 1, dim), gradH(i, j), gradV(i, j)];
            idx = idx + 1;
        end
    end
end
end