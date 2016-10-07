load('ex6data3.mat');

C = 1;
sigma = 0.1;

c_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50]';
sigma_vec = [0.01, 0.02, 0.03, 0.1, 0.3, 1, 3, 10]';

error_train = zeros(length(sigma_vec), 1);
error_val = zeros(length(sigma_vec), 1);

for i=1:size(sigma_vec, 1)
  model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma_vec(i)));
  error_train(i) = mean(double(svmPredict(model, X) ~= y));
  error_val(i) = mean(double(svmPredict(model, Xval) ~= yval));
end

plot(sigma_vec, error_train, sigma_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('sigma');
ylabel('Error');

% error_train = zeros(length(c_vec), 1);
% error_val = zeros(length(c_vec), 1);
%
% for i=1:size(c_vec, 1)
%   model = svmTrain(X, y, c_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma));
%   error_train(i) = mean(double(svmPredict(model, X) ~= y));
%   error_val(i) = mean(double(svmPredict(model, Xval) ~= yval));
% end
%
% plot(c_vec, error_train, c_vec, error_val);
% legend('Train', 'Cross Validation');
% xlabel('C');
% ylabel('Error');

fprintf('Program paused. Press enter to continue.\n');
pause;
