clear all;
clc

%% Calculating the inequality constraints for the convex set Ss in NED

% The North-East-Down coordinate system is denoted {n} where the
%       x_n axis points towards true North
%       y_n axis points towards East
%      (z_n axis points downwards normal to the Earth's surface)

% harbor_vertices_nordhus = [   0   0;      % Trondheim Hurtigruta harbor
%                      35 100;
%                     250 185;
%                     600 275;
%                     600 100;
%                       0   0];


harbor_vertices_martinsen = [   0   0;      % Trondheim Hurtigruta harbor
                               -8  22;
                               10  75;
                              750 250;
                              750  40];

harbor_vertices_hareid = [      0   0;      % Trondheim Hurtigruta harbor
                             -218  54;
                             -475 402;
                             -139 638;
                              120 348];

% Using vert2lcon() to find the linear inequality constraints defining the
% convex polygon in R^2 given its vertices.

[As,bs] = vert2lcon(harbor_vertices_martinsen)
[As_Hareid,bs_Hareid] = vert2lcon(harbor_vertices_hareid)

