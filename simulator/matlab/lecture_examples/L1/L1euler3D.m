clear
clc
%% L1: Euler angles and 3-D visualization of NED positions 
% Solve the kinematics in a real-time loop using Euler angles.
% P-controllers for surge velocity and yaw (heading) angle.                  

addpath("flypath3d_v2/")    % Add folder for 3-D visualization files

% User inputs
h = 0.1;                    % sampling time
N = 300;                    % number of samples

% Autopilot setpoints
u_d = 3;                    % surge velocity
psi_d = deg2rad(20);        % yaw angle

% model parameters
m = 1;
Iz = 1;
d_u = 1;
d_r = 1;

% controller gains
kp_u = 1;
kp_psi = 1;

% initial states
eta = [0 0 0 0 0 0]';      % eta = [x y z phi theta psi]' 
nu  = [0.5 0.1 0 0 0 0]';  % nu  = [u v w p q r]'

% allocate empty table for simulation data
simdata = zeros(N+1,1+length(eta)+length(nu)); 

%% MAIN LOOP
for i = 1:N+1
   
   t = (i-1) * h;                          % time (s)  
   
   % measurements
   psi = eta(6);
   u = nu(1);
   r = nu(6);
 
   % control laws
   tau1 = d_u * u_d - kp_u * (u - u_d);         % surge controller

   if t > 10, psi_d = deg2rad(-30); end   
   tau6 = -kp_psi * ssa( psi - psi_d );         % yaw controller

   % kinematics
   R = Rzyx(eta(4),eta(5),eta(6));
   T = Tzyx(eta(4),eta(5));
   J = [         R  zeros(3,3)
         zeros(3,3)         T ];
   
   % differential equations
   eta_dot = J * nu;
   nu_dot = [ (1/m) * (tau1 - d_u * u )
              zeros(4,1) 
              (1/Iz) * (tau6 - d_r * r) ];

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

