function imgHSV = rgb2hsv_custom(imgRGB)
% Convierte una imagen RGB a espacio de color HSV
% Basado en la teoría de la lección 5

imgRGB = double(imgRGB);
[h, w, ~] = size(imgRGB);
imgHSV = zeros(h, w, 3);

for i = 1:h
    for j = 1:w
        R = imgRGB(i, j, 1);
        G = imgRGB(i, j, 2);
        B = imgRGB(i, j, 3);

        % Valor V (intensidad)
        V = max([R, G, B]);

        % Saturación S
        minRGB = min([R, G, B]);
        if V ~= 0
            S = (V - minRGB) / V;
        else
            S = 0;
        end

        % Tono H
        if V == minRGB
            H = 0;
        else
            delta = V - minRGB;
            if V == R
                H = 60 * (G - B) / delta;
            elseif V == G
                H = 120 + 60 * (B - R) / delta;
            else % V == B
                H = 240 + 60 * (R - G) / delta;
            end
        end

        if H < 0
            H = H + 360;
        end

        imgHSV(i, j, 1) = H / 360;  % Normalizar a [0,1]
        imgHSV(i, j, 2) = S;
        imgHSV(i, j, 3) = V;
    end
end
end