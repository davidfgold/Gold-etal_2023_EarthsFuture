_your zenodo badge here_

# Gold-etal_2024_EarthsFuture

**Exploring the Spatially Compounding Multi-sectoral Drought Vulnerabilities in Colorado's West Slope River Basins**

David F. Gold<sup>1\*</sup>, Patrick M. Reed <sup>2</sup> and Rohini S. Gupta <sup>2</sup>

<sup>1 </sup> Department of Physical Geography, Faculty of Geosciences, Utrecht University, Utrecht, Netherlands
<sup>2 </sup> School of Civil and Environmental Engineering, Cornell University, Ithaca, NY

\* corresponding author:  d.f.gold@uu.nl

## Abstract
The state of Colorado’s West Slope Basins are critical headwaters of the Colorado River and play a vital role in supporting Colorado’s local economy and natural environment. However, balancing the multi-sectoral water demands in the West Slope Basins is an increasing challenge for water managers. Internal variability - irreducible uncertainty stemming from interactions across non-linear processes within the hydroclimate system - complicates future vulnerability assessments. Climate change may exacerbate drought vulnerability in the West Slope Basins, with significant streamflow declines possible by mid-century. In this work, we introduce a novel multi-site Hidden Markov Model (HMM)-based synthetic streamflow generator to create an ensemble of streamflows for all six West Slope Basins that better characterizes the region’s hydroclimate and drought extremes. We capture the effects of climate change by perturbing the HMM to generate a climate-adjusted ensemble of streamflows that reflects plausible changes in climate. We then route both ensembles of streamflows through StateMod, the state of Colorado’s water allocation model, to evaluate spatially compounding drought impacts across the West Slope basins. Our results illustrate how drought events emerging from the system’s stationary internal variability in the absence of climate change can have significant impacts that exceed extreme conditions in the historical record. Further, we find that even relatively modest levels of plausible climate changes can cause a regime shift where extreme drought impacts become routine. These results can inform future Colorado River planning efforts, and our methodology can be expanded to other snow-dominated regions that face persistent droughts.


## Journal reference
Gold, D.F, Reed, P.M. & Gupta, R.S. (In Prep).Exploring the Spatially Compounding Multi-sectoral Drought Vulnerabilities in Colorado's West Slope River Basins. Earth's Future
## Code reference
In process.  

These are generated by Zenodo automatically when conducting a release when Zenodo has been linked to your GitHub repository. The Zenodo references are built by setting the author order in order of contribution to the code using the author's GitHub user name.  This citation can, and likely should, be edited without altering the DOI.

## Data reference

### Input data
Colorado Decision Support Systems: [https://cdss.colorado.gov/modeling-data/surface-water-statemod](https://cdss.colorado.gov/modeling-data/surface-water-statemod)

### Output data


## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| StateMod | 15.0 | [https://github.com/OpenCDSS/cdss-app-statemod-fortran](https://github.com/OpenCDSS/cdss-app-statemod-fortran) | - |

## Reproduce my experiment
This experiment has three main phases. First, the multi-site HMM is fit to the historical record of streamflows in the West Slope Basins. This can be done locally on a laptop or desktop. Second, the streamflow ensembles are run through StateMod, and output is collected and compressed. This step must be done on an HPC resource. This experiment was conducted on [The Cube](https://www.cac.cornell.edu/wiki/index.php?title=THECUBE_Cluster) and [Hopper](https://www.cac.cornell.edu/wiki/index.php?title=Hopper_Cluster) clusters at Cornell University. Finally, the StateMod output is post-processed, and figures are generated. This step can be done on a laptop, but is recommended to be completed on a HPC resource. 

### 1. Fit the multi-site HMM and generate synthetic streamflow ensembles
| Script Name | Description | How to Run |
| --- | --- | --- |
| `fit_hmm.py` | Fits the HMM to 75 years of historical record and saves parameters to text files| `python3 fit_hmm.py` |
| `create_synthetic_records.py` | uses the HMM parameters to generate a baseline ensemble of annual streamflow records for each basin | `python3 create_synthetic_records.py` |
| `annual_records_to_xbm.py` | disaggregates baseline synthetic records across space and time and creates StateMod input files (xbm) | `python3 annual_records_to_xbm.py` |
| `create_synthetic_records_climate.py` | applies climate change adjustements the HMM parameters and generates an adjusted ensemble of annual streamflow records for each basin | `python3 create_synthetic_records_climate.py` |
| `annual_records_to_xbm_climate.py` | disaggregates climate adjusted synthetic records across space and time and creates StateMod input files (xbm) | `python3 annual_records_to_xbm_climate.py` |
| `shift_snowmelt_monthly.py` | shifts climate adjusted streamflow records 1 month earlier to account for changes in snowmelt timing | `python3 shift_snowmelt_monthly.py` |

### 2. Run the HMM ensembles through StateMod and compress the data output

1. Download and install StateMod from [Contributing modeling software](#contributing-modeling-software)
2. Navigate to the "fortran" directory, located within the directory you just downloaded (cdss-app-statemod-fortran/src/main/fortran)
3. Open the file called "makefile" and remove the term "-static" from lines 164 and 171
4. Compile the StateMod executable by typing: `make statemod` 
5. Download and install the StateMod input data for each West Slope basin (Upper Colorado, Gunnison, Yampa, White, and San Juan/Dolores) from CDSS [Input data](#input-data) and unzip the files
6. For each basin
   a. Navigate to the "StateMod" directory. For example, for the Gunnison basin, navigate to "gm2015_StateMod_modified/      StateMod".
   b. Create two new directories, one called "baseline_run" and one called "climate_run"
   c. Navigate to "baseline_run" and create a new directory called "generated_input_files", and inside              "generated_input_files" create a directory called "xbm"
   d. Upload the 1000 xbm files generated by the baseline HMM to the xbm directory
   e. Navigate back to the "baseline_run" directory, and create a new directory called "scenarios".
   f. Navigate into "scenarios" and upload the files in "Workflow/StateModProductionRuns" from this repository
   g. Create a Python environment using the requirements.txt file within the Workflow directory of this repository
   h. Run the scripts in the table below
   i. Navigate to the "climate_run" directory (created in step 6b) and repeat steps 6c-g, uploading the 1000 climate-adjusted xbm files in step 6d instead of the baseline scenarios
 8. Repeat step 6 for each basin  

| Script Name | Description | How to Run |
| --- | --- | --- |
| `sim_set_up.` | Creates 1000 directories, titled "S0_1" to "S999_1" and creates a symbolic link to the StateMod executable (step 4) within each directory | `./sim_set_up.sh` |
| `gen_rsp.py` | Fills in a template .rsp file (which controls StatMod runs) | `python3 gen_rsp.py` |

4. Download and unzip the output data from my experiment [Output data](#output-data)
5. Run the following scripts in the `workflow` directory to compare my outputs to those from the publication

| Script Name | Description | How to Run |
| --- | --- | --- |
| `compare.py` | Script to compare my outputs to the original | `python3 compare.py --orig /path/to/original/data.csv --new /path/to/new/data.csv` |

## Reproduce my figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication.

| Script Name | Description | How to Run |
| --- | --- | --- |
| `generate_figures.py` | Script to generate my figures | `python3 generate_figures.py -i /path/to/inputs -o /path/to/outuptdir` |
