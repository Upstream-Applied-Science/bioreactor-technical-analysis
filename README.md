# Bioreactor Technical Analysis (BioTA)
Technical analysis workflow development for yield prediction and optimisation in bioreactors for cultivated meat production.
The motivation is to build on existing approaches by extending the modelling to couple with computational fluid dynamic models 
or other sources of bioreactor performance characteristics and to utilise optimisation techniques to generate optimum yield predictions for variations in bioreactor architecture, geometry and operating conditions. 

This work has been funded by the Good Food Institute (https.gfi.org) through their 2022 RFP.

## Usage
Clone to a working directory or to a location in your python path. Future releases may be provided as pip packages.

## Release History

### v0.1.0 Initial release 
Uses existing published modelling approaches for biorector performance and cell metabolism.
Examples for basic use of yield prediction and brute force optimisation.

### v1.0.0 Optimisation workflow release
Provides:
1. Two optimisation workflows which use different system level bioreactor models to predict yield from performance characteristics generated by a CFD model
2. The system level bioreactor models
3. Templates of the CFD models used in the optimisations and additional CFD models used to assess the effects of spatial variation in dissolved oxygen concntration coupled to concentration dependent cellular uptake rate
4. Results of the optimisations

Details are provided in [engrxiv](https://engrxiv.org/preprint/view/3788).
