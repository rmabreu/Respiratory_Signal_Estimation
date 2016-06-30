% Author: Rodolfo Abreu, ISR/IST, Universidade de Lisboa, 2016

function [ E_qrs, E_wb, E_twave, dEig_qrs, dEig_wb, dEig_twave ] = ...
    kPCA_aux(cov_QRS, dim_QRS, sig_qrs, cov_WB, dim_WB, sig_wb, cov_Twave, dim_Twave, sig_twave)

[ ~, mapping_QRS_g ]   = compute_mapping_v2(cov_QRS, ...
    'KernelPCA', dim_QRS, 'gauss', sig_qrs);
[ ~, mapping_WB_g ]    = compute_mapping_v2(cov_WB, ...
    'KernelPCA', dim_WB, 'gauss', sig_wb);
[ ~, mapping_Twave_g ] = compute_mapping_v2(cov_Twave, ...
    'KernelPCA', dim_Twave, 'gauss', sig_twave);

% Entropy
E_qrs   = entropy(mapping_QRS_g.K);
E_wb    = entropy(mapping_WB_g.K);
E_twave = entropy(mapping_Twave_g.K);

% Difference between eigenvalues
eigenvalues_QRS   = 1 ./ diag(mapping_QRS_g.invsqrtL);
dEig_qrs   = eigenvalues_QRS(1) - sum(eigenvalues_QRS(2:end));

eigenvalues_WB    = 1 ./ diag(mapping_WB_g.invsqrtL);
dEig_wb    = eigenvalues_WB(1) - sum(eigenvalues_WB(2:end));

eigenvalues_Twave = 1 ./ diag(mapping_Twave_g.invsqrtL);
dEig_twave = eigenvalues_Twave(1) - sum(eigenvalues_Twave(2:end));