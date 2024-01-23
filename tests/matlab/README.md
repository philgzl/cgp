# CGP
This is a Matlab implementation of CGP, the correlation based glimpse proportion index for speech intelligibiligy prediction. The algorithm takes as input the time-aligned clean and degraded speech singals. The output of the measure is expected to have a monotonic increasing relationship with subjective speech intelligibility. 

## Installation 
Before using the CGP function, the ```utils``` folder has to be added to the Matlab search path. 

## Usage
The function ```cgp``` takes three inputs:
```MATLAB
d = cgp(x, y, fs_sig);
```

* ```x```: an array of a single-channel clean speech signal.
* ```y```: an array of a single-channel degraded speech signal.
* ```fs_sig```: the sampling rate of the input signals in ```Hz```.
* ```d```: overall CGP score. 

The input speech signals are assumed to be time-aligned and of the same length.

## References
If you use CGP, please cite the reference below:
```
[1] A. Alghamdi, L. Moen, W. Y. Chan, D. Fogerty, and J. Jensen, "Correlation based glimpse proportion index," WASPAA 2023.
```
