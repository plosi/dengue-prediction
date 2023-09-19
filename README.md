# Dengue early warning
A Machine Learning approach for prediction of dengue fever outbreaks in Peru, using climate global and regional indicators as an early warning tool for emergency preparedeness.

## Problem definition
According to WHO, as of early 2023 about half of the world's population is at risk of dengue with an estimated 100–400 million infections occurring each year. Several studies show how climate change allows Aedes aegypti mosquitos to infest new areas, as a consequence it can be expected that dengue, chikungunya, Zika and yellow fever are likely to emerge in previously uninfected areas. As an example, the recent outbreak of dengue in Madeira, Portugal, in 2012-13.

This is an exercise to understand how ML algorithms can be used to develop an early warning tool for dengue preparedness, based on available climate data.

## Data
Epidemiological data: PAHO/WHO (https://www3.paho.org/data/index.php/en/mnu-topics/indicadores-dengue-en.html)

Climate data: ERA5 (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview), NOAA (https://www.noaa.gov/)


### Data summary
- cases:      total number of dengue cases (week)
- prec:       rainfall in mm (day >> week)
- tmin:       minimum air temperature in Celsius (day >> week)
- tmax:       maximum air temperature in Celsius (day >> week)
- tmean:      average air temperature in Celsius (day >> week)
- dewmean:    average dew point in Celsius (day >> week)
- rh:         average relative humidity in %, calculated from dewmean and tmean (day >> week)
- sst:        average sea surface temperature in Celsius (day >> week)

Data available from 01-01-2014 to 31-07-2023.

## Methods
- Use of epidemiological data (total reported dengue cases) combined with regional (rainfall, air temperature and humidity) and global (SST) climate indicators
- Time-series analysis using XGBoost
- Feature engineering and hyperparameters tuning
- Forecast

The model is trained on data from 01-01-2014 to 31-12-2022. The remaining part of the dataset is used to test the model performance when predicting on unseen data.

## Results
The model shows a score of about 90%, and RMSE values less than half the standard deviation, thus can be considered a good model. The feature importance ratings show how both regional and global climate indicators play an important role in predicting the diffusion of the disease.
The model has good predicting capacity for a three months horizon.

## Limitations and way forward
- Get better/new data
- Try different gradient boosting algorithms
- Explore use of LSTM networks
- Develop a userfriendly interface for non-technical users
