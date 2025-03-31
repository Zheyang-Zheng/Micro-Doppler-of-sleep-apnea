% LOAD BINARY FILE
data_dir = 'C:\Users\ZZY\Desktop\micro-doppler\';
file_name = 'breathnorm9.dat';

fullFileName = strcat(data_dir, file_name);

fileID = fopen(fullFileName, 'r');
dataArray = textscan(fileID, '%f');
fclose(fileID);
radarData = dataArray{1};
clearvars fileID dataArray ans;

% Parameters
fc = radarData(1); % Center frequency
Tsweep = radarData(2) / 1000; % Sweep time in sec (converted from ms)
NTS = radarData(3); % Number of time samples per sweep
Bw = radarData(4); % FMCW Bandwidth
Data = radarData(5:end); % Raw data in I+j*Q format

fs = NTS / Tsweep; % Sampling frequency ADC
record_length = 10; % Length of recording in seconds
nc = record_length / Tsweep; % Number of chirps

% Reshape data into chirps and do range FFT (1st FFT)
Data_time = reshape(Data, [NTS, 2*nc]); 
Data_Rx_1 = Data_time(:, 1:2:end);
Data_Rx_2 = Data_time(:, 2:2:end);

% Range FFT (1st FFT)
tmp_Rx1 = fftshift(fft(Data_Rx_1), 1);
tmp_Rx2 = fftshift(fft(Data_Rx_2), 1);

Data_range_Rx1(1:NTS/2, :) = tmp_Rx1(NTS/2+1:NTS, :);
Data_range_Rx2(1:NTS/2, :) = tmp_Rx2(NTS/2+1:NTS, :);


% Choose target range bin for phase extraction
target_range_bin = 2; % Adjust as needed
I_signal = real(Data_range_Rx1(target_range_bin, :)); % Extract I component
Q_signal = imag(Data_range_Rx1(target_range_bin, :)); % Extract Q component

% Fit a circle to I/Q data
center_I = mean(I_signal); 
center_Q = mean(Q_signal); 

% Remove DC bias by shifting the data
I_signal = I_signal - center_I;
Q_signal = Q_signal - center_Q;

% Design a high-pass filter (cutoff ~0.1 Hz, adjust based on signal characteristics)
[b, a] = butter(4, 0.1 / (fs / 2), 'high'); 

% Apply high-pass filtering to remove slow trends
I_filtered = filter(b, a, I_signal);
Q_filtered = filter(b, a, Q_signal);

% Compute phase and unwrap
phase_signal = atan2(Q_signal, I_signal); % Compute phase
phase_unwrapped = unwrap(phase_signal); % Unwrap phase to remove jumps

% Generate time axis
time_axis = linspace(0, record_length, length(phase_unwrapped));

% Plot phase variation over time
figure;
plot(time_axis, phase_unwrapped);
xlabel('Time (s)');
ylabel('Phase (radians)');
title('Phase Variation Over Time (Rx1, Range Bin 2)');
grid on;


