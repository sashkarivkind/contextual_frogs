function [hPatch, hLine] = errorInBetween(x, y, e, varargin)
% errorInBetween plots a shaded error region (y ± e) over x.
%
%   [hPatch, hLine] = errorInBetween(x, y, e) creates a blue shaded band
%   around the data y with error e (standard deviation or standard error) and
%   plots the mean line on top.
%
%   [hPatch, hLine] = errorInBetween(x, y, e, 'Name', Value, ...) specifies
%   additional options using one or more name-value pair arguments:
%     'Color'      - Line and patch face color (default: [0 0.4470 0.7410])
%     'FaceAlpha'  - Transparency of the patch (default: 0.3)
%     'EdgeColor'  - Edge color of the patch (default: 'none')
%     'LineWidth'  - Width of the mean line (default: 2)
%
%   Example:
%       x = linspace(0,2*pi,50);
%       y = sin(x);
%       e = 0.2*ones(size(x));
%       errorInBetween(x, y, e, 'Color', 'r', 'FaceAlpha', 0.2);
%
%   Outputs:
%     hPatch - Handle to the patch object (error region)
%     hLine  - Handle to the line object (mean)

% Parse inputs
p = inputParser;
addParameter(p, 'Color', [0 0.4470 0.7410]);
addParameter(p, 'FaceAlpha', 0.3, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'EdgeColor', 'none');
addParameter(p, 'LineWidth', 2, @(x) isnumeric(x) && isscalar(x));
parse(p, varargin{:});

c  = p.Results.Color;
fa = p.Results.FaceAlpha;
ec = p.Results.EdgeColor;
lw = p.Results.LineWidth;

% Ensure column vectors
x = x(:);
y = y(:);
e = e(:);

% Build coordinates for patch
x2 = [x; flipud(x)];
inBetween = [y + e; flipud(y - e)];

% Plot shaded error region
hold on;
hPatch = patch(x2, inBetween, c, ...
               'FaceAlpha', fa, ...
               'EdgeColor', ec);
hPatch.Annotation.LegendInformation.IconDisplayStyle = 'off';
% Plot mean line
hLine = plot(x, y, 'Color', c, 'LineWidth', lw);
hold off;
end