clear
clc
%% L2: 6-DOF Equations of motion (MA = MRB and CA = 0 must be updated)
% Mathematical model of an Offshore Supply vessel (OSV)
%   eta_dot = J(eta) * nu
%   (MRB + MA) * nu_dot + ( CRB(nu) + CA(nu) + D ) * nu + G * eta = tau

addpath("flypath3d_v2/")    % Add folder for 3-D visualization files

% Simulation parameters
h = 0.1;                    % sampling time (s)
N = 800;                    % number of samples

% Autopilot setpoints
u_d = 3;                    % surge velocity (m/s)
psi_d = deg2rad(20);        % yaw angle (rad)

%% Ship model parameters
L = 83;                     % length (m)
B = 18;                     % beam (m)
T = 5;                      % draft (m)
rho = 1025;                 % density of water (kg/m3)
Cb = 0.75;                  % block coefficient: Cb = nabla / (L * B * T) 
nabla = Cb * L * B * T;     % volume displacement(m3) 
m = rho * nabla             % mass (kg)
r_bg = [-0.5 0 -1]';        % location of the CG with respect to the CO

Cw = 0.8;                   % waterplane area coefficient: Cw = Awp/(L * B)
Awp = Cw * B * L;           % waterplane area
KB = (1/3) * (5*T/2 - nabla/Awp)                          % Eq. (4.38)
k_munro_smith =  (6 * Cw^3) / ( (1+Cw) * (1+2*Cw))        % Eq. (4.37)
r_bb = [-0.5 0 T-KB]';      % location of the CB with respect to the CO
BG = r_bb(3) - r_bg(3)      % vertical distance between CG and CB

I_T = k_munro_smith * (B^3 * L) / 12;   % transverse moment of inertia                  
I_L = 0.7 * (L^3 * B) / 12;             % longitudinal moment of inertia
BM_T = I_T / nabla;
BM_L = I_L / nabla;
GM_T = BM_T - BG                        % should be between 0.5 and 1.5 m
GM_L = BM_L - BG

% MRB and MA matrices
R44 = 0.35 * B;          % radius of gyration in roll, see Eq.(4.77)-(4.78)
R55 = 0.25 * L;          % radius of gyration in pitch
R66 = 0.25 * L;          % radius of gyration in yaw
nu2 = [0 0 0]';
MRB = rbody(m,R44,R55,R66,nu2,r_bg')     % computes MRB in the CG
MA = MRB;                                 % UPDATE FORMULA FOR ADDED MASS!
M = MRB + MA;
Minv = inv(M);

% G matrix
LCF = -0.5;                   % x-distance from the CO to the center of Awp
r_bp = [0 0 0]';                           % compute G in the CO
G = Gmtrx(nabla,Awp,GM_T,GM_L,LCF,r_bp);

% D matrix
T1 = 10;                % time constants for linear damping (s)
T2 = 10;
T6 = 1;
zeta4 = 0.15;           % relative damping ratio in roll
zeta5 = 0.3;            % relative damping ratio in pitch
D = Dmtrx([T1, T2, T6],[zeta4,zeta5],MRB,MA,G);

% Surge: PI controller -- Closed-loop system is of 2nd order
wn_u = 0.5;                % closed-loop natural frequency in surge
kp_u = M(1,1) * 2 * wn_u;  % M11 * u_dot + kp_u * u + ki_u int(u) ) = 0 
ki_u = M(1,1) * wn_u^2;    % u_dot + 2 * zeta * wn_u * u + wn_u^2 * int(u) = 0
z_u = 0;                   % initialization of integral state 

% Yaw: PID controller -- SISO PID pole-placement Algorithm 15.1
wn_psi = 0.5;                     % closed-loop natural frequency in yaw
kp_psi = M(6,6) * wn_psi^2;    
kd_psi = M(6,6) * 2 * wn_psi;
ki_psi = (wn_psi / 10) * kp_psi;
z_psi = 0;                         % initialization of integral state 

% Initial states
eta = [0 0 0 deg2rad(5) deg2rad(1) 0]';    % eta = [x y z phi theta psi]' 
nu  = [0.5 0.1 0 0 0 0]';                  % nu  = [u v w p q r]'

% Allocate empty table for simulation data
simdata = zeros(N+1,1+length(eta)+length(nu)); 

%% MAIN LOOP
for i = 1:N+1
   
   t = (i-1) * h;                               % time (s)  
   
   % Measurements
   psi = eta(6);
   u = nu(1);
   p = nu(4);
   q = nu(5);
   r = nu(6);
   nu2 = [p, q, r]';                             % angular velocity vector
 
   % Surge: PI speed controller
   e_u = u - u_d;
   tau1 = -kp_u * e_u - ki_u * z_u;                          
   
   % Yaw: PID controller
   if t > 20, psi_d = deg2rad(-30); end
   e_psi = ssa( psi - psi_d );                    % UPDATE REFERENCE MODEL!
   tau6 = -kp_psi * e_psi - kd_psi * r - ki_psi * z_psi;   

   % Kinematics
   R = Rzyx(eta(4),eta(5),eta(6));
   T = Tzyx(eta(4),eta(5));
   J = [         R  zeros(3,3)
         zeros(3,3)         T ];
   
   % Rigid-body kinetics, computes MRB and CRB in the CG
   [MRB,CRB] = rbody(m,R44,R55,R66,nu2,r_bg'); 

   % Hydrodynamic added mass
   MA = MRB;                               % UPDATE FORMULA FOR ADDED MASS!
   CA = zeros(6,6);

   % Differential equations
   eta_dot = J * nu;
   tau = [ tau1 0 0 0 0 tau6 ]';
   nu_dot = Minv * (tau - (CRB + CA + D) * nu - G * eta);

   % Store simulation data in a table   
   simdata(i,:) = [t eta' nu']; 

   % Euler's method (k+1)
   eta = eta + h * eta_dot;
   nu = nu + h * nu_dot;
   z_u = z_u + h * e_u;             % velocity tracking error
   z_psi = z_psi + h * e_psi;       % heading angle tracking error
   
end

%% PLOTS
t     = simdata(:,1);  

x     = simdata(:,2); 
y     = simdata(:,3); 
z     = simdata(:,4); 
phi   = ssa(simdata(:,5)); 
theta = ssa(simdata(:,6)); 
psi   = ssa(simdata(:,7)); 

u     = simdata(:,8);        
v     = simdata(:,9); 
w     = simdata(:,10); 
p     = simdata(:,11);        
q     = simdata(:,12); 
r     = simdata(:,13); 

U = sqrt( u.^2 + v.^2 );     % speed
beta_c = atan(v./u);         % crab angle
chi = psi + beta_c;          % course angle

%% Create objects for 3-D visualization 
% create object 1: ship (ship1.mat)
new_object('flypath3d_v2/ship1.mat',[x,y,z,phi,theta,psi],...
'model','royalNavy2.mat','scale',0.15,...
'edge',[0 0 0],'face',[0 0 0],'alpha',1,...
'path','on','pathcolor',[.89 .0 .27],'pathwidth',1);

% create object 2: Small ship representing the desired path (ship2.mat)
%new_object('flypath3d_v2/ship2.mat',[x-5,y-5,z,phi,theta,psi],...
%'model','royalNavy2.mat','scale',0.05,...
%'edge',[0 0 0],'face',[0 0 0],'alpha',1,...
%'path','on','pathcolor',[.0 .89 .27],'pathwidth',1);

% Plot trajectories 
%flypath('ship1.mat','ship2.mat',...
flypath('ship1.mat',...    
'animate','on','step',2,...
'axis','on','axiscolor',[0 0 0],'color',[1 1 1],...
'font','Georgia','fontsize',12,...
'view',[-45 25],'window',[900 900],...
'xlim', [min(y)-10,max(y)+10],... 
'ylim', [min(x)-10,max(x)+10], ...
'zlim', [-2,10]); 

set(findall(gcf,'type','line'),'linewidth',2)

%% Position and Euler angle plots
figure(2); 
figure(gcf)
subplot(321),plot(y,x)
xlabel('East (m)')
ylabel('North (m)')
title('North-East positions (m)'),grid
subplot(322),plot(t,z)
xlabel('time (s)'),title('Down position (m)'),grid
subplot(312),plot(t,rad2deg(phi),t,rad2deg(theta))
xlabel('time (s)'),title('Roll and pitch angles (deg)'),grid
legend('Roll angle (deg)','Pitch angle (deg)')
subplot(313),plot(t,rad2deg(psi),t,rad2deg(chi),t,rad2deg(beta_c))
xlabel('time (s)'),title('Heading and course angles (deg)'),grid
legend('Yaw angle (deg)','Course angle (deg)','Crab angle (deg)')

set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','text'),'FontSize',14)
set(findall(gcf,'type','legend'),'FontSize',14)

%% Velocity plots
figure(3); 
figure(gcf)
subplot(311),plot(t,U)
xlabel('time (s)'),title('Speed (m/s)'),grid
subplot(312),plot(t,u,t,v,t,w)
xlabel('time (s)'),title('Linear velocities (m/s)'),grid
legend('u (m/s)','v (m/s)','w (m/s)')
subplot(313),plot(t,rad2deg(p),t,rad2deg(q),t,rad2deg(r))
xlabel('time (s)'),title('Angular velocities (deg/s)'),grid
legend('p (deg/s)','q (deg/s)','r (deg/s)')

set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','text'),'FontSize',14)
set(findall(gcf,'type','legend'),'FontSize',14)

