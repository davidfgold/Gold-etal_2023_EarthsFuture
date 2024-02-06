#%%
import numpy as np
from hmmlearn import hmm

def fit_HMM(annual_Q_h_path, start_year):
        """
        Fits an HMM to the historical record and returns parameters

        :param annual_Q_h_path:                        string, path to location of historical data
        :param start_year:                             float, 1st year of the historical data to include for fitting HMM

        :returns:
                -dry_state_means:                      an array of dry state means for each basin
                -wet_state_means:                      an array of wet state means for each basin
                -covariance_matrix_dry:                the covariance matrix in the dry state
                -covariance_matrix_wet:                the covariance matrix in the wet state
                -transition_matrix:                    the transition matrix from one state to another
                -hidden_states:                        an array of hidden states during the historical period
        """
        # load data
        AnnualQ_h_all = np.loadtxt(annual_Q_h_path, delimiter=',', skiprows=1)

        logAnnualQ_h = np.log(AnnualQ_h_all + 1)  # add 1 because some sites have 0 flow

        nBasins = 5

        # fit multi-site HMM to approximately last 2/3 of historical record
        hmm_model = hmm.GMMHMM(n_components=2, n_iter=1000, covariance_type='full').fit(logAnnualQ_h[start_year::, :])

        # Pull out some model parameters
        mus = np.array(hmm_model.means_)
        transition_matrix = np.array(hmm_model.transmat_)
        hidden_states = hmm_model.predict(logAnnualQ_h)

        # Dry state doesn't always come first,but we want it to be, so flip if it isn't
        if mus[0][0][0] > mus[1][0][0]:
                mus = np.flipud(mus)
                transition_matrix = np.fliplr(np.flipud(transition_matrix))
                covariance_matrix_dry = hmm_model.covars_[[1]].reshape(nBasins, nBasins)
                covariance_matrix_wet = hmm_model.covars_[[0]].reshape(nBasins, nBasins)
                hidden_states = 1 - hidden_states
        else:
                covariance_matrix_dry = hmm_model.covars_[[0]].reshape(nBasins, nBasins)
                covariance_matrix_wet = hmm_model.covars_[[1]].reshape(nBasins, nBasins)

        # Redefine variables
        dry_state_means = mus[0, :]
        wet_state_means = mus[1, :]

        return dry_state_means, wet_state_means, covariance_matrix_dry, covariance_matrix_wet, \
               transition_matrix, hidden_states

#%%
# Fit the HMM with 75 years of historical data
LR_dry_means, LR_wet_means, LR_covariance_dry, LR_covariance_wet, LR_transition_matrix, LR_hidden_states = \
        fit_HMM('../../historical_data/all_basins.csv', 30)

#%% save the HMM parameters
np.savetxt('HMM_parameters/LR/dry_state_means.txt',LR_dry_means)
np.savetxt('HMM_parameters/LR/wet_state_means.txt',LR_wet_means)
np.savetxt('HMM_parameters/LR/covariance_matrix_dry.txt',LR_covariance_dry)
np.savetxt('HMM_parameters/LR/covariance_matrix_wet.txt',LR_covariance_wet)
np.savetxt('HMM_parameters/LR/transition_matrix.txt', LR_transition_matrix)
#%%
np.savetxt('HMM_parameters/LR/hist_hidden_states.txt', LR_hidden_states)
#%%
# Fit the HMM with 730 years of historical data
SR_dry_means, SR_wet_means, SR_covariance_dry, SR_covariance_wet, SR_transition_matrix, SR_hidden_states = \
        fit_HMM('../../historical_data/all_basins.csv', 75)
#%%
# save the HMM parameters
np.savetxt('HMM_parameters/SR/dry_state_means.txt',SR_dry_means)
np.savetxt('HMM_parameters/SR/wet_state_means.txt',SR_wet_means)
np.savetxt('HMM_parameters/SR/covariance_matrix_dry.txt',SR_covariance_dry)
np.savetxt('HMM_parameters/SR/covariance_matrix_wet.txt',SR_covariance_wet)
np.savetxt('HMM_parameters/SR/transition_matrix.txt', SR_transition_matrix)
#%%
np.savetxt('HMM_parameters/SR/hist_hidden_states.txt', SR_hidden_states)