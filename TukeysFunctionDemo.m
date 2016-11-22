clearvars; close all; clc;

res = randn(1,1000)*30;

%normalize residuals
resMAD = res./(1.4826.*mad(res(:),1));
sat_c=4.685;

%tukey's fucntion
y = f(resMAD,sat_c);

%analytical derivative
tu=resMAD.*(1-(resMAD./sat_c).^2).^2;
y_prim= tu.*bsxfun(@lt,abs(resMAD),sat_c);

%numerical derivative
dt= 0.1;
y_prim_num = zeros(1,numel(resMAD));
for i=1:numel(resMAD)
    %y_prim_num(i) = ( f(x(i)+ dt) - f(x(i)) )/dt;
    y_prim_num(i) = ( f(resMAD(i)+ dt,sat_c) - f(resMAD(i) - dt,sat_c) )/ (2*dt);
end

subplot(3,1,1); scatter(resMAD,y); grid on; title('Tukeys Biweight Function');
subplot(3,1,2); scatter(resMAD,y_prim); grid on; title('Analytical Derivative');
subplot(3,1,3); scatter(resMAD,y_prim_num); grid on; title('Numerical Derivative');

function y = f(resMAD,sat_c)
%tukey's biweight function
y = ((sat_c.^2)/6) * (1 - (1-(resMAD./sat_c).^2).^3);
y(abs(resMAD)>sat_c) = (sat_c.^2)/6;
end