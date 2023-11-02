clear
clc
%% L1: Euler angles
% Solve the kinematics in a real-time loop using Euler angles.
% P-controllers for surge velocity and yaw (heading) angle. 

% User inputs
h = 0.1;                    % sampling time
N = 150;                    % number of samples

% Autopilot setpoints
u_d = 3;                    % surge velocity
psi_d = deg2rad(10);        % yaw angle

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
phi   = rad2deg( ssa(simdata(:,5)) ); 
theta = rad2deg( ssa(simdata(:,6)) ); 
psi   = rad2deg( ssa(simdata(:,7)) ); 

u     = simdata(:,8);        
v     = simdata(:,9); 
w     = simdata(:,10); 
p     = rad2deg( simdata(:,11) );        
q     = rad2deg( simdata(:,12) ); 
r     = rad2deg( simdata(:,13) ); 

U = sqrt( u.^2 + v.^2 );                % speed
beta_c = rad2deg( atan(v./u) );         % crab angle
chi = psi + beta_c;                     % course angle

%% Position and Euler angle plots
figure(1); 
figure(gcf)
subplot(321),plot(y,x)
xlabel('East (m)')
ylabel('North (m)')
title('North-East positions (m)'),grid
subplot(322),plot(t,z)
xlabel('time (s)'),title('Down position (m)'),grid
subplot(312),plot(t,phi,t,theta)
xlabel('time (s)'),title('Roll and pitch angles (deg)'),grid
legend('Roll angle (deg)','Pitch angle (deg)')
subplot(313),plot(t,psi,t,chi,t,beta_c)
xlabel('time (s)'),title('Heading and course angles (deg)'),grid
legend('Yaw angle (deg)','Course angle (deg)','Crab angle (deg)')

set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','text'),'FontSize',14)
set(findall(gcf,'type','legend'),'FontSize',14)

%% Velocity plots
figure(2); 
figure(gcf)
subplot(311),plot(t,U)
xlabel('time (s)'),title('Speed (m/s)'),grid
subplot(312),plot(t,u,t,v,t,w)
xlabel('time (s)'),title('Linear velocities (m/s)'),grid
legend('u (m/s)','v (m/s)','w (m/s)')
subplot(313),plot(t,p,t,q,t,r)
xlabel('time (s)'),title('Angular velocities (deg/s)'),grid
legend('p (deg/s)','q (deg/s)','r (deg/s)')

set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','text'),'FontSize',14)
set(findall(gcf,'type','legend'),'FontSize',14)
