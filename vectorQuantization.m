function confMat = confusionMatrix(predicciones, etiquetas, numClases)
% Calcula la matriz de confusión para visualizar resultados

confMat = zeros(numClases, numClases);
n = length(predicciones);

for i = 1:n
    confMat(etiquetas(i), predicciones(i)) = confMat(etiquetas(i), predicciones(i)) + 1;
end

% Mostrar matriz de confusión
for i = 1:numClases
    for j = 1:numClases
        fprintf('%6d ', confMat(i, j));
    end
    fprintf('\n');
end
end