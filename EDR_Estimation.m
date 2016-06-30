% EDR_Estimation() - estimates a surrogate of the respiratory signal from
%                    single-lead ECG data using 8 different methods found
%                    in the literature (HRV, AM1, AM2, QRS-AM, ENV, PCA, kPCA and EMD)
% Usage:
%    >> EDR = EDR_Estimation(dataset, filters_ecg, filters_resp, clean_peaks, plt, nb_plts)
%
% Inputs:
%   dataset      - input data structure with the following mandatory fields:
%                    1) data -> EEG signal (each row represent an EEG channel)
%                    2) ecg -> ECG signal (row vector)
%                    3) srate [double] -> sampling rate
%                    4) Kp -> R peak annotations
%   filters_ecg  - low and high cutoff values of the band-pass filtering of ECG data {[ low_cutoff high_cutoff ]}
%   filters_resp - low and high cutoff values of the hypothesized respiratory signal
%                  (typically 0.2-04 Hz) {[ low_cutoff high_cutoff ]}
%   clean_peaks  - 1/0 = do/do not perform additional steps for false
%                  positive and false negative R peaks based on beat morphology
%   plt          - 1/0 = do/don't display the results
%   nb_plts      - number of subplots within each plot for displaying the IMFs obtained from the EMD
%
% Outputs:
%   EDR [cell]   - respiratory signals estimated using all different methods
%                  (.name -> name of the method, .signal -> the respiratory trace)
%
% Author: Rodolfo Abreu, ISR/IST, Universidade de Lisboa, 2016

function EDR = EDR_Estimation(dataset, filters_ecg, filters_resp, clean_peaks, plt, nb_plts)

fs = dataset.srate; R = dataset.Kp;

m_R = (min(diff(R)) / dataset.srate) * 1000;
m_R_smp = mean(diff(R)); % average R-to-R peak interval

% ECG filtering
ecg = eegfilt(dataset.ecg, fs, filters_ecg(1), 0);
ecg = eegfilt(ecg, fs, 0, filters_ecg(2));

% ECG baseline removal
medfilt_200 = 0.2 * fs; medfilt_600 = 0.6 * fs;
ecg_200ms = medfilt1(ecg, medfilt_200);
ecg_baseline = medfilt1(ecg_200ms, medfilt_600);
ecg_rm_baseline = ecg - ecg_baseline;

lim_inf_ms = -100;
if m_R < 4 * abs(lim_inf_ms)
    lim_sup_ms = lim_inf_ms + (4 * abs(lim_inf_ms));
else
    lim_sup_ms = lim_inf_ms + m_R;
end

L_Resp = length(ecg);

% EDR from the ECG envelope using the Hilbert transform
ecg_env = abs(hilbert(ecg(R)));
EDR_Env = spline(R, ecg_env, 1:L_Resp);
EDR_Env = lowpass_filter(EDR_Env, 0.5, fs);

% EDR from RR peak differences
HRV = diff(R);
EDR_HRV = spline(R(1:end-1), HRV, 1:L_Resp);
EDR_HRV = lowpass_filter(EDR_HRV, 0.5, fs);

% EDR from R peaks amplitude (R - S waves)
S = zeros(1, length(R));
win_80ms = 0.08 * fs;
for i = 1:length(R)
    W = ecg(R(i):R(i) + win_80ms);
    [ ~, m ] = min(W);
    S(i) = R(i) + m;
end

RAmp = R - S;
EDR_RAmp = spline(R, RAmp, 1:L_Resp);
EDR_RAmp = lowpass_filter(EDR_RAmp, 0.5, fs);

% EDR from R peak amplitudes after baseline removal
EDR_RAmp_rm = spline(R, ecg_rm_baseline(R), 1:L_Resp);
EDR_RAmp_rm = lowpass_filter(EDR_RAmp_rm, 0.5, fs);

% EDR from QRS area after baseline removal
win_100ms = 0.1 * fs;
if R(1) - ceil(win_100ms / 2) < 0, R(1) = []; end
if R(end) + ceil(win_100ms / 2) > length(ecg), R(end) = []; end
QRS = zeros(1, length(R));

for i = 1:length(R)
    W = ecg_rm_baseline(R(i) - ceil(win_100ms / 2):R(i) + ceil(win_100ms / 2));
    QRS(i) = sum(abs(W));
end

EDR_QRS = spline(R, QRS, 1:L_Resp);
EDR_QRS = lowpass_filter(EDR_QRS, 0.5, fs);

% ERD FROM PCA - FEATURES

if R(1) - ceil(m_R_smp * 0.5) < 0, R(1) = []; end
if R(end) + win_80ms > length(ecg), R(end) = []; end

R_80ms_m = R - ceil(m_R_smp * 0.5); R_80ms_M = R + win_80ms;
R_Tw_m = R_80ms_M; R_Tw_M = R_80ms_m(2:end);
m_RTw = ceil(mean(R_Tw_M - R_Tw_m(1:end - 1)));
R_Tw_M = horzcat(R_Tw_M, R_Tw_M(end) + m_RTw);
win_RTw = median(R_Tw_M - R_Tw_m); R_Tw_M = R_Tw_m + win_RTw;

if R(1) - ceil((m_R_smp * 0.9) / 2) < 0, R(1) = []; end
if R(end) + ceil((m_R_smp * 0.9) / 2) > length(ecg), R(end) = []; end

R_QRS_m = R - ceil(win_100ms / 2); R_QRS_M = R + ceil(win_100ms / 2);
R_WB_m = R - ceil((m_R_smp * 0.9) / 2); R_WB_M = R + ceil((m_R_smp * 0.9) / 2);

l_QRS = length(R_QRS_m(1):R_QRS_M(1)); QRS_complex = zeros(l_QRS, length(R));
l_WB = length(R_WB_m(1):R_WB_M(1)); WB = zeros(l_WB, length(R));
l_Twave = win_RTw + 1; Twave = zeros(l_Twave, length(R));

for i = 1:length(R)
    QRS_complex(:, i) = ecg(R_QRS_m(i):R_QRS_M(i)); % QRS complexes
    WB(:, i) = ecg(R_WB_m(i):R_WB_M(i)); % whole ECG beats
    Twave(:, i) = ecg(R_Tw_m(i):R_Tw_M(i)); % T wave
end

cov_QRS = cov(QRS_complex); cov_WB = cov(WB); cov_Twave = cov(Twave);

[ coeff_QRS, ~, ~ ]   = pcacov(cov_QRS);
[ coeff_WB, ~, ~ ]    = pcacov(cov_WB);
[ coeff_Twave, ~, ~ ] = pcacov(cov_Twave);

nb_pcs = 3;
EDR_PCA_QRS   = zeros(nb_pcs, L_Resp);
EDR_PCA_WB    = zeros(nb_pcs, L_Resp);
EDR_PCA_Twave = zeros(nb_pcs, L_Resp);

for i = 1:nb_pcs
    EDR_PCA_QRS(i, :)   = lowpass_filter(spline(R, coeff_QRS(:, i), 1:L_Resp), 0.5, fs);
    EDR_PCA_WB(i, :)    = lowpass_filter(spline(R, coeff_WB(:, i), 1:L_Resp), 0.5, fs);
    EDR_PCA_Twave(i, :) = lowpass_filter(spline(R, coeff_Twave(:, i), 1:L_Resp), 0.5, fs);
end

% ERD FROM KERNEL PCA - FEATURES

% Polynomial Kernel
pk = [ 2 3 ];

% Gaussian Kernel
sig_QRS         = l_QRS .* mean(var(QRS_complex));
sig_QRS_range   = linspace(sig_QRS / 100, sig_QRS * 100, 1000);

sig_WB          = l_WB .* mean(var(WB));
sig_WB_range    = linspace(sig_WB / 100, sig_WB * 100, 1000);

sig_Twave       = l_Twave .* mean(var(Twave));
sig_Twave_range = linspace(sig_Twave / 100, sig_Twave * 100, 1000);

% Estimation of the number of dimensions
dim_QRS   = round(intrinsic_dim(cov_QRS, 'MLE'));
dim_WB    = round(intrinsic_dim(cov_WB, 'MLE'));
dim_Twave = round(intrinsic_dim(cov_Twave, 'MLE'));

% Kernel PCA
[ mapped_QRS_p2, ~ ]   = compute_mapping(cov_QRS, 'KernelPCA', dim_QRS, 'poly', 1, pk(1));
[ mapped_QRS_p3, ~ ]   = compute_mapping(cov_QRS, 'KernelPCA', dim_QRS, 'poly', 1, pk(2));

[ mapped_WB_p2, ~ ]    = compute_mapping(cov_WB, 'KernelPCA', dim_WB, 'poly', 1, pk(1));
[ mapped_WB_p3, ~ ]    = compute_mapping(cov_WB, 'KernelPCA', dim_WB, 'poly', 1, pk(2));

[ mapped_Twave_p2, ~ ] = compute_mapping(cov_Twave, 'KernelPCA', dim_Twave, 'poly', 1, pk(1));
[ mapped_Twave_p3, ~ ] = compute_mapping(cov_Twave, 'KernelPCA', dim_Twave, 'poly', 1, pk(2));

E_QRS   = zeros(1, length(sig_QRS_range));
E_WB    = zeros(1, length(sig_WB_range));
E_Twave = zeros(1, length(sig_Twave_range));

dEig_QRS   = zeros(1, length(sig_QRS_range));
dEig_WB    = zeros(1, length(sig_WB_range));
dEig_Twave = zeros(1, length(sig_Twave_range));

% Standard Deviation - Serial
for i = 1:length(sig_QRS_range)
    [ E_qrs, E_wb, E_twave, dEig_qrs, dEig_wb, dEig_twave ] = ...
        kPCA_aux(cov_QRS, dim_QRS, sig_QRS_range(i), cov_WB, dim_WB, ...
        sig_WB_range(i), cov_Twave, dim_Twave, sig_Twave_range(i));
    
    E_QRS(i) = E_qrs; E_WB(i) = E_wb; E_Twave(i) = E_twave;
    
    dEig_QRS(i) = dEig_qrs; dEig_WB(i) = dEig_wb; dEig_Twave(i) = dEig_twave;
end

% Maximizing Entropy
[ ~, M_QRS ] = max(E_QRS); [ ~, M_WB ] = max(E_WB); [ ~, M_Twave ] = max(E_Twave);

% Maximizing the difference between eigenvalues
[ ~, m_QRS ] = max(dEig_QRS); [ ~, m_WB ] = max(dEig_WB); [ ~, m_Twave ] = max(dEig_Twave);

sig = [ M_QRS, M_WB, M_Twave, m_QRS, m_WB, m_Twave ];

% Kernel PCA with a Gaussian kernel that maximizes the Entropy
[ mapped_QRS_gE, ~ ]      = compute_mapping...
    (cov_QRS, 'KernelPCA', dim_QRS, 'gauss', sig(1));

[ mapped_WB_gE, ~ ]       = compute_mapping...
    (cov_WB, 'KernelPCA', dim_WB, 'gauss', sig(2));

[ mapped_Twave_gE, ~ ]    = compute_mapping...
    (cov_Twave, 'KernelPCA', dim_Twave, 'gauss', sig(3));

% Kernel PCA with a Gaussian kernel that maximizes the Differences between eigenvalues
[ mapped_QRS_gDEig, ~ ]   = compute_mapping...
    (cov_QRS, 'KernelPCA', dim_QRS, 'gauss', sig(4));

[ mapped_WB_gDEig, ~ ]    = compute_mapping...
    (cov_WB, 'KernelPCA', dim_WB, 'gauss', sig(5));

[ mapped_Twave_gDEig, ~ ] = compute_mapping...
    (cov_Twave, 'KernelPCA', dim_Twave, 'gauss', sig(6));

EDR_kPCA_QRS_p2 = zeros(dim_QRS, L_Resp); EDR_kPCA_QRS_p3 = zeros(dim_QRS, L_Resp);
EDR_kPCA_QRS_gE = zeros(dim_QRS, L_Resp); EDR_kPCA_QRS_gDEig = zeros(dim_QRS, L_Resp);

EDR_kPCA_WB_p2 = zeros(dim_WB, L_Resp); EDR_kPCA_WB_p3 = zeros(dim_WB, L_Resp);
EDR_kPCA_WB_gE = zeros(dim_WB, L_Resp); EDR_kPCA_WB_gDEig  = zeros(dim_WB, L_Resp);

EDR_kPCA_Twave_p2 = zeros(dim_Twave, L_Resp); EDR_kPCA_Twave_p3 = zeros(dim_Twave, L_Resp);
EDR_kPCA_Twave_gE = zeros(dim_Twave, L_Resp); EDR_kPCA_Twave_gDEig  = zeros(dim_Twave, L_Resp);

for i = 1:dim_QRS
    EDR_kPCA_QRS_p2(i, :)    = lowpass_filter(spline(R, mapped_QRS_p2(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_QRS_p3(i, :)    = lowpass_filter(spline(R, mapped_QRS_p3(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_QRS_gE(i, :)    = lowpass_filter(spline(R, mapped_QRS_gE(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_QRS_gDEig(i, :) = lowpass_filter(spline(R, mapped_QRS_gDEig(:, i), 1:L_Resp), 0.5, fs);
end

for i = 1:dim_WB
    EDR_kPCA_WB_p2(i, :)    = lowpass_filter(spline(R, mapped_WB_p2(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_WB_p3(i, :)    = lowpass_filter(spline(R, mapped_WB_p3(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_WB_gE(i, :)    = lowpass_filter(spline(R, mapped_WB_gE(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_WB_gDEig(i, :) = lowpass_filter(spline(R, mapped_WB_gDEig(:, i), 1:L_Resp), 0.5, fs);
end

for i = 1:dim_Twave
    EDR_kPCA_Twave_p2(i, :)    = lowpass_filter(spline(R, mapped_Twave_p2(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_Twave_p3(i, :)    = lowpass_filter(spline(R, mapped_Twave_p3(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_Twave_gE(i, :)    = lowpass_filter(spline(R, mapped_Twave_gE(:, i), 1:L_Resp), 0.5, fs);
    EDR_kPCA_Twave_gDEig(i, :) = lowpass_filter(spline(R, mapped_Twave_gDEig(:, i), 1:L_Resp), 0.5, fs);
end

% EDR FROM EMPIRICAL MODE DECOMPOSITION (EMD)

IMF = EMD(ecg, 50, 50, 1);
selected_IMF = IMF_selection(IMF, filters_resp, plt, nb_plts);
EDR_EMD = IMF(:, selected_IMF)';

methods = { 'Env', 'HRV', 'RAmp', 'RAmp_rm', 'QRS', 'PCA_QRS', 'PCA_WB', ...
    'PCA_Twave', 'kPCA_QRS_p2', 'kPCA_QRS_p3', 'kPCA_QRS_gE', 'kPCA_QRS_gDEig', ...
    'kPCA_WB_p2', 'kPCA_WB_p3', 'kPCA_WB_gE', 'kPCA_WB_gDEig', 'kPCA_Twave_p2', ...
    'kPCA_Twave_p3', 'kPCA_Twave_gE', 'kPCA_Twave_gDEig', 'EMD' };

EDR(1).name  = 'Env';              EDR(1).signal  = EDR_Env;
EDR(2).name  = 'HRV';              EDR(2).signal  = EDR_HRV;
EDR(3).name  = 'RAmp';             EDR(3).signal  = EDR_RAmp;
EDR(4).name  = 'RAmp_rm';          EDR(4).signal  = EDR_RAmp_rm;
EDR(5).name  = 'QRS';              EDR(5).signal  = EDR_QRS;
EDR(6).name  = 'PCA_QRS';          EDR(6).signal  = EDR_PCA_QRS;
EDR(7).name  = 'PCA_WB';           EDR(7).signal  = EDR_PCA_WB;
EDR(8).name  = 'PCA_Twave';        EDR(8).signal  = EDR_PCA_Twave;
EDR(9).name  = 'kPCA_QRS_p2';      EDR(9).signal  = EDR_kPCA_QRS_p2;
EDR(10).name = 'kPCA_QRS_p3';      EDR(10).signal = EDR_kPCA_QRS_p3;
EDR(11).name = 'kPCA_QRS_gE';      EDR(11).signal = EDR_kPCA_QRS_gE;
EDR(12).name = 'kPCA_QRS_gDEig';   EDR(12).signal = EDR_kPCA_QRS_gDEig;
EDR(13).name = 'kPCA_WB_p2';       EDR(13).signal = EDR_kPCA_WB_p2;
EDR(14).name = 'kPCA_WB_p3';       EDR(14).signal = EDR_kPCA_WB_p3;
EDR(15).name = 'kPCA_WB_gE';       EDR(15).signal = EDR_kPCA_WB_gE;
EDR(16).name = 'kPCA_WB_gDEig';    EDR(16).signal = EDR_kPCA_WB_gDEig;
EDR(17).name = 'kPCA_Twave_p2';    EDR(17).signal = EDR_kPCA_Twave_p2;
EDR(18).name = 'kPCA_Twave_p3';    EDR(18).signal = EDR_kPCA_Twave_p3;
EDR(19).name = 'kPCA_Twave_gE';    EDR(19).signal = EDR_kPCA_Twave_gE;
EDR(20).name = 'kPCA_Twave_gDEig'; EDR(20).signal = EDR_kPCA_Twave_gDEig;
EDR(21).name = 'EMD';              EDR(21).signal = EDR_EMD;

if plt
    % Clean R peaks
    if clean_peaks
        ecg_norm = ecg ./ max(ecg);
        figure, plot(ecg_norm), hold on, plot(R_copy, C_WB, '-ro');
        plot([ 1 length(ecg) ], [ threshold threshold ], 'k');
    end
    
    % Window tests
    R_80ms = R + win_80ms;
    R_50ms_m = R - ceil(win_100ms / 2); R_50ms_M = R + ceil(win_100ms / 2);
    
    figure('Name', 'ECG_f'), hold on, plot(ecg)
    plot(R, ecg(R), 'ro', 'MarkerFaceColor', 'r')
    axis('tight'),
    line([ R_50ms_m; R_50ms_m ], [ min(ecg) max(ecg) ], 'Color', 'g');
    line([ R_50ms_M; R_50ms_M ], [ min(ecg) max(ecg) ], 'Color', 'g');
    % line([ R_WB; R_WB ], [ min(ecg) max(ecg) ], 'Color', 'k');
    hold off
    
    % R_WB = R + m_R;
    R_WB_m = R - ceil((m_R_smp * 0.9) / 2); R_WB_M = R + ceil((m_R_smp * 0.9) / 2);
    figure('Name', 'ECG_f'), hold on, plot(ecg)
    plot(R, ecg(R), 'ro', 'MarkerFaceColor', 'r')
    axis('tight'),
    line([ R_WB_m; R_WB_m ], [ min(ecg) max(ecg) ], 'Color', 'k');
    line([ R_WB_M; R_WB_M ], [ min(ecg) max(ecg) ], 'Color', 'k');
    hold off
    
    win_70ms = 0.07 * fs;
    R_80ms_m = R - ceil(m_R_smp * 0.55); R_80ms_M = R + win_80ms;
    R_Tw_m = R_80ms_M; R_Tw_M = R_80ms_m(2:end);
    m_RTw = ceil(mean(R_Tw_M - R_Tw_m(1:end - 1)));
    R_Tw_M = horzcat(R_Tw_M, R_Tw_M(end) + m_RTw);
    figure('Name', 'ECG_f'), hold on, plot(ecg)
    plot(R, ecg(R), 'ro', 'MarkerFaceColor', 'r')
    axis('tight'),
    line([ R_Tw_m; R_Tw_m ], [ min(ecg) max(ecg) ], 'Color', 'g');
    line([ R_Tw_M; R_Tw_M ], [ min(ecg) max(ecg) ], 'Color', 'g');
    hold off
    
    figure('Name', 'ECG_f'), hold on, plot(ecg)
    plot(R, ecg(R), 'ro', 'MarkerFaceColor', 'r')
    plot(S, ecg(S), 'ko', 'MarkerFaceColor', 'k')
    line([ R_80ms; R_80ms ], [ min(ecg) max(ecg) ], 'Color', 'g');
    axis('tight')
    hold off
end