% IMF_Selection() - identifies the most accurate surrogate of the
%                   respiratory signal within the EMD-computed IMFs based 
%                   on a frequency-based criterion
% Usage:
%    >> selected_IMF = IMF_selection(IMF, filters_imf, plt, nb_plts)
%
% Inputs:
%   IMF         - input IMFs already computed using an EMD routine [ time
%              samples x number of IMFs ]
%   filters_imf - low and high cutoff values of the hypothesized respiratory signal
%                  (typically 0.2-04 Hz) {[ low_cutoff high_cutoff ]}
%   plt         - 1/0 = do/don't display the results
%   nb_plts     - number of subplots within each plot for displaying the IMFs obtained from the EMD
%
% Outputs:
%   selected_IMF - IMF index of the most accurate surrogate of the
%                  respiratory signal
%
% Author: Rodolfo Abreu, ISR/IST, Universidade de Lisboa, 2016

function selected_IMF = IMF_selection(IMF, filters_imf, plt, nb_plts)
    
% Frequency domain "parameters"
Fs = dataset.srate; L = length(IMF);
NFFT = 2^nextpow2(L); % Next power of 2 from length of y
f = Fs/2 * linspace(0, 1, NFFT/2 + 1);

f1_s = find(f <= filters_imf(1)); f1_samples = f1_s(end);
f2_s = find(f <= filters_imf(2)) + 1; f2_samples = f2_s(end);

f_s = horzcat(filters_imf(1), filters_imf(2));
f_samples = horzcat(f1_samples, f2_samples);

f_10Hz = find(f <= 10); f_10Hz_samp = f_10Hz(end); F_s = 1:f_10Hz_samp;
F_noresp = F_s; F_noresp(f1_samples:f2_samples) = 0;
F_noresp(F_noresp == 0) = [];

P = zeros(1, size(IMF, 2));
P_noresp = zeros(1, size(IMF, 2));

count = 1;
figure('Name', 'IMF'); hold on
for jj = 1:size(IMF, 2)
    y = IMF(:, jj);
    Y = fft(y, NFFT) / L;
    PowerSpectrum = 2 * abs(Y(1:NFFT/2 + 1));
    PowerSpectrum = PowerSpectrum ./ max(PowerSpectrum);
    
    if plt
        T = (1:L) ./ Fs;
        if count <= nb_plts
            subplot(nb_plts, 2, (2 * count) - 1), plot(T, IMF(:, jj)), axis('tight')
            ylabel([ 'IMF', num2str(jj) ])
            
            subplot(nb_plts, 2, (2 * count)), plot(f(1:f_10Hz_samp), PowerSpectrum(1:f_10Hz_samp)), axis('tight')
            ylabel('|Y(f)|')
            line([ f_s; f_s ], [ min(PowerSpectrum), max(PowerSpectrum) ], ...
                'Color', 'red', 'LineWidth', 2);
            
            count = count + 1;
        else
            count = 1;
            
            % Y-axis label for power spectrum plot
            text(5, -0.18, 'Frequency [Hz]', 'HorizontalAlignment', 'Center');
            % X-axis label for power spectrum plot
            text(5, 3.88, 'Power Spectrum Density', 'HorizontalAlignment', 'Center');
            % Y-axis label for the respiratory signal
            text(-8, -0.18, 'Time [s]', 'HorizontalAlignment', 'Center');
            % X-axis label for the respiratory signal
            text(-8, 3.88, 'Intrinsic Mode Function (IMF)', 'HorizontalAlignment', 'Center');
            
            hold off
            figure('Name', 'IMF'), hold on
            
            subplot(nb_plts, 2, (2 * count) - 1), plot(T, IMF(:, jj)), axis('tight')
            ylabel([ 'IMF', num2str(jj) ])
            
            subplot(nb_plts, 2, (2 * count)), plot(f(1:f_10Hz_samp), PowerSpectrum(1:f_10Hz_samp)), axis('tight')
            ylabel('|Y(f)|')
            line([ f_s; f_s ], [ min(PowerSpectrum), max(PowerSpectrum) ], ...
                'Color', 'red', 'LineWidth', 2);
            
            count = count + 1;
        end
    end
    P(jj) = sum(PowerSpectrum(f1_samples:f2_samples));
    P_noresp(jj) = sum(PowerSpectrum(F_noresp));
end
hold off

Ratio = P ./ P_noresp;
[ ~, index ] = sort(Ratio, 'descend'); selected_IMF = index(1:2);

if plt
    figure('Name', 'PSD', 'Color', 'white')
    bar(P, 0.5), colormap('winter')
    set(gca, 'XLim', [ 0, size(IMF, 2) + 1 ]);
    ylabel('IMFs')
    title('Power Spectrum Density (PSD)')
    
    figure('Name', 'Ratio', 'Color', 'white')
    bar(Ratio, 0.5), colormap('winter')
    set(gca, 'XLim', [ 0, size(IMF, 2) + 1 ]);
    ylabel('IMFs')
    title('Ratio between PSD [0.2; 0.4]Hz and PSD [0; 0.2[ U ]0.4; 5]Hz')
end       
            
            
            
