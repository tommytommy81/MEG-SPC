# MEG-SPC

Python based code for detection and clustering spikes for MEG data. Example of one cluster of spikes detected on magnetometers: 

![Output template example](https://github.com/vagechirkov/MEG-SPC/blob/master/Example%20output%20plots/magnetometers/9_temp.png)

## Documentation
###Preprocassing steps:
* Maxwell filter and movement compensation
* Filter 1Hz
* Plot power spectral density (PSD)
* Manually select the file for the further analysis
* Plot ICA components
* Manually select bad components
* Apply ICA

###Detection:
* Copy files *config.params* and *meg_306.prb* in the *circus* folder
* Run Spyking Circus
* Plot clusters

Full pipeline with example dataset: *Code/main.ipynb*

## Requirements


* __python 3.6__
* __Anaconda 3__
* __Inkscape__

Package | Version
------------ | -------------
mne | 0.18.2
ipypb | 0.5.2
spyking-circus | 0.8.3
svgutils | 0.3.1

