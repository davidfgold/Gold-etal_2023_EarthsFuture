import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import os
os.chdir('C:/Users/dgold/Dropbox/Postdoc/IM3/Colorado/InternalVariabilityPaper/paper_code/final_data_analysis')
#%%
def calc_plot_percentile_shortage(record_type, percentile, ax, legend, hist):
    print('../../Results/Shortage/cu/updated_cm_' + record_type + '_no_export.csv')
    # read data
    cm_data = np.loadtxt('../../Results/Shortage/cu/updated_cm_' + record_type + '_no_export.csv', delimiter=',') * 1233.48
    gm_data = np.loadtxt('../../Results/Shortage/cu/updated_gm_' + record_type + '_no_export.csv', delimiter=',') * 1233.48
    ym_data = np.loadtxt('../../Results/Shortage/cu/updated_ym_' + record_type + '_no_export.csv', delimiter=',') * 1233.48
    wm_data = np.loadtxt('../../Results/Shortage/cu/updated_wm_' + record_type + '_no_export.csv', delimiter=',') * 1233.48
    sj_data = np.loadtxt('../../Results/Shortage/cu/updated_sj_' + record_type + '_no_export.csv', delimiter=',') * 1233.48

    # get percentile of interest across all realizations
    cm_percentiles = np.zeros([1000])
    gm_percentiles = np.zeros([1000])
    ym_percentiles = np.zeros([1000])
    wm_percentiles = np.zeros([1000])
    sj_percentiles = np.zeros([len(sj_data[0,:])])

    for r in range(1000):
        cm_percentiles[r] = np.percentile(cm_data[:, r], percentile)
        gm_percentiles[r] = np.percentile(gm_data[:, r], percentile)
        ym_percentiles[r] = np.percentile(ym_data[:, r], percentile)
        wm_percentiles[r] = np.percentile(wm_data[:, r], percentile)

    for r in range(len(sj_data[0,:])):
        sj_percentiles[r] = np.percentile(sj_data[:, r], percentile)

    # now calculate the percentiles across the 1000 realizations to make plot
    cm_realizations = np.zeros(100)
    gm_realizations = np.zeros(100)
    ym_realizations = np.zeros(100)
    wm_realizations = np.zeros(100)
    sj_realizations = np.zeros(100)

    for p in range(1, 101):
        cm_realizations[p - 1] = np.percentile(cm_percentiles, p)
        gm_realizations[p - 1] = np.percentile(gm_percentiles, p)
        ym_realizations[p - 1] = np.percentile(ym_percentiles, p)
        wm_realizations[p - 1] = np.percentile(wm_percentiles, p)
        sj_realizations[p - 1] = np.percentile(sj_percentiles, p)

    # sum all and sort
    total = cm_realizations + gm_realizations + ym_realizations + wm_realizations + sj_realizations


    # create plot input
    base = sj_realizations/1000000
    second = base + wm_realizations/1000000
    third = second + ym_realizations/1000000
    fourth = third + gm_realizations/1000000
    fifth = fourth + cm_realizations/1000000

    print(max(fifth))

    ax.fill_between(np.arange(len(base)), np.zeros(100), base, alpha=.85, color='#335c67',edgecolor='none')
    ax.fill_between(np.arange(len(base)), base, second, alpha=.85, color='#fff3b0',edgecolor='none')
    ax.fill_between(np.arange(len(base)), second, third, alpha=.85, color='#e09f3e',edgecolor='none')
    ax.fill_between(np.arange(len(base)), third, fourth, alpha=.85, color='#9e2a2b',edgecolor='none')
    ax.fill_between(np.arange(len(base)), fourth, fifth, alpha=.85, color='#540b0e',edgecolor='none')


    ax.plot(np.arange(len(base)), np.ones(len(base))*hist, linestyle='--', color='lightblue')
    ax.set_ylabel('Annual Consumptive Use Shortage (Million $m^3$)')
    ax.set_xlabel('Realization Percentile')
    ax.set_ylim([0, 1000])
    ax.set_xlim(0,100)
    if record_type == 'base':
        ax.set_title('Baseline Ensemble\n'+str(percentile) + 'th percentile ')
    else:
        ax.set_title('Climate-Adjusted Ensemble\n' + str(percentile) + 'th percentile ')
    if legend:
        ax.legend(['Southwest', 'White', 'Yampa', 'Gunnison', 'Upper Colorado'])

#%% Diff Plot
def diff_plot(percentile, ax, ylabels):
    # read data
    cm_base = np.loadtxt('../../Results/Shortage/cu/updated_cm_baseline_no_export.csv', delimiter=',') * 1233.48
    gm_base = np.loadtxt('../../Results/Shortage/cu/gm_base.csv', delimiter=',') * 1233.48
    ym_base = np.loadtxt('../../Results/Shortage/cu/ym_base.csv', delimiter=',') * 1233.48
    wm_base = np.loadtxt('../../Results/Shortage/cu/wm_base.csv', delimiter=',') * 1233.48
    sj_base = np.loadtxt('../../Results/Shortage/cu/updated_sj_baseline_no_export.csv', delimiter=',') * 1233.48

    # calc percentiles
    cm_b_p = np.mean(np.percentile(cm_base, percentile, axis=1))/1000000000
    gm_b_p = np.mean(np.percentile(gm_base, percentile, axis=0))/1000000000
    ym_b_p = np.mean(np.percentile(ym_base, percentile, axis=0))/1000000000
    wm_b_p = np.mean(np.percentile(wm_base, percentile, axis=0))/1000000000
    sj_b_p = np.mean(np.percentile(sj_base, percentile, axis=1))/1000000000

    # read climate
    cm_clim = np.loadtxt('../../Results/Shortage/cu/updated_cm_mod_climate_no_export.csv', delimiter=',') * 1233.48
    gm_clim = np.loadtxt('../../Results/Shortage/cu/updated_gm_mod_climate.csv', delimiter=',') * 1233.48
    ym_clim = np.loadtxt('../../Results/Shortage/cu/updated_ym_mod_climate_no_export.csv', delimiter=',') * 1233.48
    wm_clim = np.loadtxt('../../Results/Shortage/cu/updated_wm_mod_climate_no_export.csv', delimiter=',') * 1233.48
    sj_clim = np.loadtxt('../../Results/Shortage/cu/updated_sj_mod_climate_no_export.csv', delimiter=',') * 1233.48

    # calc percentiles
    cm_c_p = np.mean(np.percentile(cm_clim, percentile, axis=1))/1000000000
    gm_c_p = np.mean(np.percentile(gm_clim, percentile, axis=1))/1000000000
    ym_c_p = np.mean(np.percentile(ym_clim, percentile, axis=1))/1000000000
    wm_c_p = np.mean(np.percentile(wm_clim, percentile, axis=1))/1000000000
    sj_c_p = np.mean(np.percentile(sj_clim, percentile, axis=1))/1000000000

    ax.plot([cm_b_p, cm_c_p], [5, 5], color='#540b0e', zorder=-1)
    ax.scatter(cm_b_p, 5, color='#540b0e', edgecolor='none', s=100)
    ax.scatter(cm_c_p, 5, color='w', edgecolor='#540b0e', s=100)

    ax.plot([gm_b_p, gm_c_p], [4, 4], color='#9e2a2b', zorder=-5)
    ax.scatter(gm_b_p, 4, color='#9e2a2b', edgecolor='none', s=100)
    ax.scatter(gm_c_p, 4, color='w', edgecolor='#9e2a2b', s=100)

    ax.plot([ym_b_p, ym_c_p], [3, 3], color='#e09f3e', zorder=-9)
    ax.scatter(ym_b_p, 3, color='#e09f3e', edgecolor='none', s=100)
    ax.scatter(ym_c_p, 3, color='w', edgecolor='#e09f3e', s=100)

    ax.plot([wm_b_p, wm_c_p], [2, 2], color='#fff3b0', zorder=-12)
    ax.scatter(wm_b_p, 2, color='#fff3b0', edgecolor='none', s=100)
    ax.scatter(wm_c_p, 2, color='w', edgecolor='#fff3b0', s=100)

    ax.plot([sj_b_p, sj_c_p], [1, 1], color='#335c67', zorder=-15)
    ax.scatter(sj_b_p, 1, color='#335c67', edgecolor='none', s=100)
    ax.scatter(sj_c_p, 1, color='w', edgecolor='#335c67', s=100)

    ax.set_ylim([0.5,5.5])
    ax.set_yticks([1,2,3,4,5])
    ax.set_xlabel('Consumptive Use Shortage (maf)')
    ax.set_xlim([0,.35])
    if ylabels:
        ax.set_yticklabels(['Southwest', 'White', 'Yampa', 'Gunnison', 'Upper Colorado'])
    else:
        ax.set_yticklabels(['', '', '', '', ''])


#%%
fig, axes = plt.subplots(2,3, figsize=(12, 8))
calc_plot_percentile_shortage('base', 50, axes.flatten()[0], False, 116*1.233)
calc_plot_percentile_shortage('base', 90, axes.flatten()[1], False, 252.5*1.233)
calc_plot_percentile_shortage('base', 99, axes.flatten()[2], False, 457.8*1.233)
#calc_plot_percentile_shortage('AdjustedClimate', 50, axes.flatten()[3], False,116.0*1.233)
#calc_plot_percentile_shortage('AdjustedClimate', 90, axes.flatten()[4], False,252.5*1.233)
calc_plot_percentile_shortage('AdjustedClimate', 99, axes.flatten()[5], False, 457.8*1.233)

#axes.flatten()[0].plot(np.arange(100))
#diff_plot(50, axes.flatten()[6], True)
#diff_plot(90, axes.flatten()[7], False)
#diff_plot(99, axes.flatten()[8], False)

plt.tight_layout()
plt.show()
#plt.savefig('../../Figures/InitialSubmissionFigures/AdjustedClimate/Seminar/Shortage.png')

#%%
fig, axes = plt.subplots(1,3, figsize = (12,3))
diff_plot(50, axes[0], True)
diff_plot(90, axes[1], False)
diff_plot(99, axes[2], False)
plt.tight_layout()
plt.show()

#%%
record_type = 'mod_climate'
percentile = 99
cm_data = np.loadtxt('../../Results/Shortage/cu/cm_' + record_type + '_no_export.csv', delimiter=',')
gm_data = np.loadtxt('../../Results/Shortage/cu/gm_' + record_type + '.csv', delimiter=',')
ym_data = np.loadtxt('../../Results/Shortage/cu/ym_' + record_type + '.csv', delimiter=',')
wm_data = np.loadtxt('../../Results/Shortage/cu/wm_' + record_type + '.csv', delimiter=',')
sj_data = np.loadtxt('../../Results/Shortage/cu/sj_' + record_type + '_no_export.csv', delimiter=',')

# get percentile of interest across all realizations
cm_percentiles = np.zeros([1000])
gm_percentiles = np.zeros([1000])
ym_percentiles = np.zeros([1000])
wm_percentiles = np.zeros([1000])
sj_percentiles = np.zeros([972])

for r in range(1000):
    cm_percentiles[r] = np.percentile(cm_data[:, r], percentile)
    gm_percentiles[r] = np.percentile(gm_data[:, r], percentile)
    ym_percentiles[r] = np.percentile(ym_data[:, r], percentile)
    wm_percentiles[r] = np.percentile(wm_data[:, r], percentile)

for r in range(len(sj_data[0,:])):
    sj_percentiles[r] = np.percentile(sj_data[:, r], percentile)

# now calculate the percentiles across the 1000 realizations to make plot
cm_realizations = np.zeros(99)
gm_realizations = np.zeros(99)
ym_realizations = np.zeros(99)
wm_realizations = np.zeros(99)
sj_realizations = np.zeros(99)

for p in range(1, 100):
    cm_realizations[p - 1] = np.percentile(cm_percentiles, p)
    gm_realizations[p - 1] = np.percentile(gm_percentiles, p)
    ym_realizations[p - 1] = np.percentile(ym_percentiles, p)
    wm_realizations[p - 1] = np.percentile(wm_percentiles, p)
    sj_realizations[p - 1] = np.percentile(sj_percentiles, p)

# sum all and sort
total = cm_realizations + gm_realizations + ym_realizations + wm_realizations + sj_realizations

# create plot input
base = sj_realizations/1000
second = base + wm_realizations/1000
third = second + ym_realizations/1000
fourth = third + gm_realizations/1000
fifth = fourth + cm_realizations/1000

fig = plt.figure()
ax = fig.gca()

ax.fill_between(np.arange(len(base)), np.zeros(99), base, alpha=.85, color='#335c67',edgecolor='none')
ax.fill_between(np.arange(len(base)), base, second, alpha=.85, color='#fff3b0',edgecolor='none')
ax.fill_between(np.arange(len(base)), second, third, alpha=.85, color='#e09f3e',edgecolor='none')
ax.fill_between(np.arange(len(base)), third, fourth, alpha=.85, color='#9e2a2b',edgecolor='none')
ax.fill_between(np.arange(len(base)), fourth, fifth, alpha=.85, color='#540b0e',edgecolor='none')


#%% Load baseline ensemble
cm_base = np.loadtxt('../../Results/Shortage/cu/cm_climate_no_export.csv', delimiter=',')
gm_base = np.loadtxt('../../Results/Shortage/cu/gm_climate.csv', delimiter=',')
ym_base = np.loadtxt('../../Results/Shortage/cu/ym_climate.csv', delimiter=',')
wm_base = np.loadtxt('../../Results/Shortage/cu/wm_climate.csv', delimiter=',') #FIX THIS
sj_base = np.loadtxt('../../Results/Shortage/cu/sj_climate_no_export.csv', delimiter=',')

#%% calculate percentiles

cm_percentiles = np.zeros([99, 1000])
gm_percentiles = np.zeros([99, 1000])
ym_percentiles = np.zeros([99, 1000])
wm_percentiles = np.zeros([99, 1000])
sj_percentiles = np.zeros([99, 1000])

for r in range(1000):
    for p in range(1, 100):
        cm_percentiles[p-1, r] = np.percentile(cm_base[:,r], p)
        gm_percentiles[p-1, r] = np.percentile(gm_base[:,r], p)
        ym_percentiles[p-1, r] = np.percentile(ym_base[:,r], p)
        wm_percentiles[p-1, r] = np.percentile(wm_base[:,r], p)
        sj_percentiles[p-1, r] = np.percentile(sj_base[:,r], p)

#%%
cm_medians = np.zeros(99)
gm_medians = np.zeros(99)
ym_medians = np.zeros(99)
wm_medians = np.zeros(99)
sj_medians = np.zeros(99)

for p in range(1, 100):
    cm_medians[p-1] = np.percentile(cm_percentiles[50,:], p)
    gm_medians[p-1] = np.percentile(gm_percentiles[50,:], p)
    ym_medians[p-1] = np.percentile(ym_percentiles[50,:], p)
    wm_medians[p-1] = np.percentile(wm_percentiles[50,:], p)
    sj_medians[p-1] = np.percentile(sj_percentiles[50,:], p)

#%%
total_med = cm_medians + gm_medians + ym_medians + wm_medians + sj_medians
base = sj_medians/1000
second = base + wm_medians/1000
third = second + gm_medians/1000
fourth = third + ym_medians/1000
fifth = fourth + cm_medians/1000

#%%
fig = plt.figure()
plt.fill_between(np.arange(len(base)),np.zeros(len(base)), base, alpha = .85, color='#335c67', edgecolor='none')
plt.fill_between(np.arange(len(base)), base, second, alpha = .85, color='#fff3b0', edgecolor='none')
plt.fill_between(np.arange(len(base)), second, third, alpha = .85, color='#e09f3e', edgecolor='none')
plt.fill_between(np.arange(len(base)), third, fourth, alpha = .85, color='#9e2a2b', edgecolor='none')
plt.fill_between(np.arange(len(base)), fourth, fifth, alpha = .85, color='#540b0e', edgecolor='none')
plt.show()