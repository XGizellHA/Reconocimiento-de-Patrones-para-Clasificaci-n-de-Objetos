function plotConfusionMatrix(confMat, clases, titulo)
% Grafica una matriz de confusión como figura de MATLAB
% confMat: matriz de confusión
% clases: cell array con nombres de las clases
% titulo: título de la figura

numClases = length(clases);
figure('Name', titulo, 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);

% Calcular porcentajes
confMatPercent = confMat ./ sum(confMat, 2) * 100;
confMatPercent(isnan(confMatPercent)) = 0;

% Crear la matriz de confusión
imagesc(confMat);
colormap(parula);
colorbar;

% Configurar ejes
xticks(1:numClases);
yticks(1:numClases);
xticklabels(clases);
yticklabels(clases);
xtickangle(45);
set(gca, 'FontSize', 10);

% Título y etiquetas
title(titulo, 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Predicción', 'FontSize', 12);
ylabel('Valor Real', 'FontSize', 12);

% Agregar valores en las celdas
for i = 1:numClases
    for j = 1:numClases
        if confMat(i, j) > 0
            % Color del texto: blanco si fondo oscuro, negro si fondo claro
            if confMat(i, j) > max(confMat(:)) / 2
                textColor = 'w';
            else
                textColor = 'k';
            end
            text(j, i, sprintf('%d\n(%.1f%%)', confMat(i, j), confMatPercent(i, j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 9, ...
                'Color', textColor, ...
                'FontWeight', 'bold');
        end
    end
end

% Ajustar límites de color
caxis([0, max(confMat(:))]);

% Agregar cuadrícula
grid on;
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.3);

% Guardar la figura automáticamente
saveas(gcf, fullfile('results', [strrep(titulo, ' ', '_'), '.png']));
end
