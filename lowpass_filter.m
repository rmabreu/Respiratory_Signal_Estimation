% lowpass_filters() - performs a second order low-pass filter
% Usage:
%    >> signal_filtered = lowpass_filter(signal, cutoff_freq, fs)
%
% Inputs:
%   signal      - input signal
%   cutoff_freq - high cutoff value for the low-pass filter
%   fs          - sampling frequency
%
% Outputs:
%   signal_filtered - filtered signal
%
% Author: Rodolfo Abreu, ISR/IST, Universidade de Lisboa, 2016

function signal_filtered = lowpass_filter(signal, cutoff_freq, fs)

order = 2;

[coef_num, coef_denom] =  butter(order, ((cutoff_freq * 2) / fs), 'low');

signal_filtered = filter(coef_num, coef_denom, signal);

return;