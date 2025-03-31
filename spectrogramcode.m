% LOAD BINARY FILE
data_dir = 'C:\Users\ZZY\Desktop\micro-doppler\';
file_name = 'pendulum5.dat';

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

% Reshape data into chirps for dual channels
Data_time = reshape(Data, [NTS, 2 * nc]);

Data_Rx_1 = Data_time(:, 1:2:end);
Data_Rx_2 = Data_time(:, 2:2:end);

% Range FFT (1st FFT)
tmp_Rx1 = fftshift(fft(Data_Rx_1), 1);
tmp_Rx2 = fftshift(fft(Data_Rx_2), 1);

Data_range_Rx1(1:NTS/2, :) = tmp_Rx1(NTS/2+1:NTS, :);
Data_range_Rx2(1:NTS/2, :) = tmp_Rx2(NTS/2+1:NTS, :);

% MTI (Moving Target Indicator) Filtering
ns = size(Data_range_Rx1, 2) - 1;
Data_range_Rx1_MTI = zeros(size(Data_range_Rx1, 1), ns);
Data_range_Rx2_MTI = zeros(size(Data_range_Rx2, 1), ns);

[b, a] = butter(4, 0.005, 'high'); % High-pass IIR filter
for k_ind = 1:size(Data_range_Rx1_MTI, 1)
    Data_range_Rx1_MTI(k_ind, 1:ns) = filter(b, a, Data_range_Rx1(k_ind, 1:ns));
    Data_range_Rx2_MTI(k_ind, 1:ns) = filter(b, a, Data_range_Rx2(k_ind, 1:ns));
end

% Remove first range bin due to filtering artifacts
Data_range_Rx1_MTI = Data_range_Rx1_MTI(2:end, :);
Data_range_Rx2_MTI = Data_range_Rx2_MTI(2:end, :);

Data_range_Rx1 = Data_range_Rx1(2:end, :);
Data_range_Rx2 = Data_range_Rx2(2:end, :);

% Compute Range and Time Axes
freq = (0:ns-1) * fs / (2 * ns); % Frequency axis
range_axis = (freq * 3e8 * Tsweep) / (2 * Bw); % Range axis in meters
time_axis_mti = linspace(0, record_length, size(Data_range_Rx1_MTI, 2)); % Time axis in seconds

% Plot Range-Time Heatmap
figure;
colormap(jet);
imagesc(time_axis_mti, range_axis, 20*log10(abs(Data_range_Rx1_MTI)));
xlabel('Time (sec)');
ylabel('Range (m)');
title('Range-Time Heatmap with MTI');

% Spectrogram Generation
% Select a range bin containing the target (range resolution*bin)
target_range_bin = 3;
Data_target = Data_range_Rx1_MTI(target_range_bin, :); % Extract data for spectrogram

% Spectrogram Parameters
window_size = 256; % Window size for better frequency resolution
overlap = 230; % 90% overlap
nfft = 1024; % Higher number of FFT points for better frequency resolution

% Generate Spectrogram
[s, f, t_raw] = spectrogram(Data_target, window_size, overlap, nfft, fs, 'yaxis');

% Correct the time axis for 10-second recording
t = t_raw * (record_length / max(t_raw)); % Scale time to match 10 seconds

% Shift Doppler frequencies to center 0 Hz
f_shifted = f - fs / 2; % Center frequency range
s_shifted = fftshift(s, 1); % Shift spectrogram data

% Doppler Velocity Scaling
lambda = 3e8 / fc; % Wavelength in meters
doppler_velocity = f_shifted * lambda / 2; % Doppler velocity in m/s

% Plot Doppler vs. Time Spectrogram (Frequency)
figure;
imagesc(t, f_shifted, 20*log10(abs(s_shifted)));
colormap(jet);
colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Doppler vs. Time Spectrogram (Frequency)');
set(gca, 'YDir', 'normal');

% Plot Doppler vs. Time Spectrogram (Velocity)
figure;
imagesc(t, doppler_velocity, 20*log10(abs(s_shifted)));
colormap(jet);
colorbar;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('Doppler vs. Time Spectrogram (Velocity)');
set(gca, 'YDir', 'normal');

figure;
imagesc(t, f_shifted, 20*log10(abs(s_shifted)));
colormap(gray); % 设为灰度
colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('');
set(gca, 'YDir', 'normal');
axis off;             % 去掉坐标轴
set(gca, 'Units', 'normalized', 'Position', [0 0 1 1]);
set(gca, 'LooseInset', get(gca,'TightInset'));
set(gca, 'Visible', 'off');
colorbar('off');      % 去掉色标

% [filepath, name, ext] = fileparts(file_name);
% new_filename = [name, '.png'];
% saveas(gcf, fullfile('C:\Users\ZZY\Desktop\dopplerdata\traning\apnea', new_filename));


%individual action
% 
% % User-defined parameters
% start_time = 2;      % Starting time in seconds
% duration = 2;        % Duration of each slice in seconds
% break_time = 0.5;      % Break time between slices in seconds
% 
% % Time-slicing parameters
% total_time = t(end); % Total available time in spectrogram
% current_time = start_time; % Initialize current time
% slice_idx = 1;             % Slice counter
% 
% % Loop through and slice the spectrogram
% while current_time + duration <= total_time
%     % Find indices corresponding to the desired time interval
%     start_idx = find(t >= current_time, 1, 'first');
%     end_idx = find(t >= (current_time + duration), 1, 'first');
%     
%     % Extract the time slice from the spectrogram
%     s_slice = s(:, start_idx:end_idx);
%     t_slice = t(start_idx:end_idx); % Corresponding time axis for this slice
%     
%     % Plot the sliced spectrogram
%     figure;
%     imagesc(t_slice, doppler_velocity, 20*log10(abs(s_slice)));
%     colormap(jet);
%     colorbar;
%     xlabel('Time (s)');
%     ylabel('Velocity (m/s)');
%     title(['Spectrogram Slice ' num2str(slice_idx)]);
%     set(gca, 'YDir', 'normal'); % Flip Y-axis to show low frequencies at bottom
%     
%     % Increment time and slice counter
%     current_time = current_time + duration + break_time;
%     slice_idx = slice_idx + 1;
% end
