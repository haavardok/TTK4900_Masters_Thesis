clear
clc
%% L1: 6-DOF Equations of motion (no damping and gravity)
% eta_dot = J(eta) * nu
% MRB * nu_dot + CRB(nu) * nu = tau
%
% NB! Since the model has no damping and gravity/buoyancy terms, the heave,
% roll and pitch equations will give you a non-physical behaviour.
% We will add these terms later.

addpath(genpath('flypath3d_v2'))    % Add folder for 3-D visualization files

% User inputs
h = 0.1;                    % sampling time
N = 300;                    % number of samples

% Autopilot setpoints
u_d = 3;                    % surge velocity
psi_d = deg2rad(20);        % yaw angle

% model parameters
r_bg = [0.1 0 0.1]';        % location of the CG with respect to the CO
R44 = 1;                    % radius of gyration in roll
R55 = 2;                    % radius of gyration in pitch
R66 = 0.5;                  % radius of gyration in yaw
m = 10;                     % mass
I_g = m * diag( [R44^2 R55^2 R66^2] ); % inertia tensor about the CG

% controller gains
kp_u = m;                       % m (u_dot + kp_u * u) = 0 

w_psi = 1;
kp_psi = I_g(3,3) * w_psi^2;    % Iz (psi_ddot + 2 w_psi psi_dot + w_psi^2) = 0
kd_psi = I_g(3,3) * 2 * w_psi;

% initial states
eta = [0 0 0 0 0 0]';      % eta = [x y z phi theta psi]' 
nu  = [0.5 0.1 0 0 0 0]';  % nu  = [u v w p q r]'

% allocate empty table for simulation data
simdata = zeros(N+1,1+length(eta)+length(nu)); 

%% MAIN LOOP
for i = 1:N+1
   
   t = (i-1) * h;                               % time (s)  
   
   % measurements
   psi = eta(6);
   u = nu(1);
   p = nu(4);
   q = nu(5);
   r = nu(6);
   nu2 = [p, q, r]';                              % angular velocity vector
 
   % control laws
   tau1 = -kp_u * (u - u_d);                           % surge: P controller
   
   if t > 10, psi_d = deg2rad(-30); end
   tau6 = -kp_psi * ssa( psi - psi_d ) - kd_psi * r;   % yaw: PD controller

   % kinematics
   R = Rzyx(eta(4),eta(5),eta(6));
   T = Tzyx(eta(4),eta(5));
   J = [         R  zeros(3,3)
         zeros(3,3)         T ];
   
   % rigid-body kinetics
   [MRB,CRB] = rbody(m,R44,R55,R66,nu2,r_bg'); % computes MRB and CRB in the CG
   
   % differential equations
   eta_dot = J * nu;
   tau = [ tau1 0 0 0 0 tau6 ]';
   nu_dot = inv(MRB) * (tau - CRB * nu );

   % store simulation data in a table   
   simdata(i,:) = [t eta' nu']; 

   % Euler's method (k+1)
   eta = eta + h * eta_dot;
   nu = nu + h * nu_dot;
   
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
new_object('flypath3d_v2/ship2.mat',[x-5,y-5,z,phi,theta,psi],...
'model','royalNavy2.mat','scale',0.05,...
'edge',[0 0 0],'face',[0 0 0],'alpha',1,...
'path','on','pathcolor',[.0 .89 .27],'pathwidth',1);

% Plot trajectories 
%flypath('ship1.mat','ship2.mat',...
flypath('ship1.mat','ship2.mat',...
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

