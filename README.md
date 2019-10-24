# MEG-SPC

Python based code for detection and clustering spikes for MEG data. Example of one cluster of spikes detected on magnetometers: 

![Output template example](https://github.com/vagechirkov/MEG-SPC/blob/master/Example%20output%20plots/magnetometers/9_temp.png)

## Documentation
Steps:
* Maxwell filter and movement compensation
* Filter 1Hz
* Plot power spectral density (PSD)
* Manually select the file for the further analysis
* Plot ICA components
* Manually select bad components
* Run Spyking Circus
* Plot clusters


## Requirements


python 3.6
Anaconda 3
Inkscape

Package | Version
------------ | -------------

mne | 0.18.2
ipypb | 0.5.2
spyking-circus | 0.8.3
svgutils | 0.3.1

