import sys
import os
import subprocess 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math 
import privateLocs

import biota.models as biomodels

import torch

from ax.service.managed_loop import optimize

# Update path to include location of M-Star binaries
sys.path.insert(0, privateLocs.libpath)
import mstar

# M-Star Path setup				
install_path = os.path.dirname(privateLocs.binpath)				
solver_exe = os.path.join(install_path, "mstar-cfd-mgpu")

# Checkout M-Star license
mstar.CheckOutLicense()

# Load template M-Star file
# This is pre-prepared in M-Star 
m=mstar.Load('MStar_templates/20kl_STR_optimisation_2.msb')


# Function to set up M-Star with prescribed parameters and generate performance characteristics
def getBioreactorMetrics(inputs):
    """
    Function called with bioreactor parameters, returns performance characteristics from M-Star simulation.

    Parameters
    ----------
    inputs: Tuple
        Operating parameters
        inputs[0]: Impeller RPM
        inputs[1]: Sparging Rate
        inputs[2]: Impeller Diameter
        inputs[3]: Impeller Blade Width

    Returns
    -------
    Tuple of the following
        kla_O2:     oxygen mass transfer coefficient
        kla_CO2:    carbon dioxide mass transfer coefficient
        tau_m:      mixing time
        maxEps:     maximum value of time avergaed eddy dissipation rate
        meanEps:    time and space averaged eddy dissipation rate
        gasVol:     biomodels state immersed bubble volume
    """

    # Set location to run M-Star
    loc = 'runDir'
    
    # Clean up run location directory if it already exists
    if os.path.isdir(loc):
        os.system('rm -r '+loc)
    
    # Make new run location
    os.makedirs(loc)

    # Set operating parameters and dimensions according to inputs
    m.Get("Moving Body").Get("Rotation Speed").Value = str(inputs[0])    
    m.Get("Bubbles").Get("Volume Feed Rate Expr").Value = str(inputs[1])
    m.Get("Rushton_1").Get("Diameter").Value = inputs[2]
    m.Get("Rushton_1_1").Get("Diameter").Value = inputs[2]
    m.Get("Rushton_1").Get("Hub Diameter").Value = inputs[2] - 0.2
    m.Get("Rushton_1_1").Get("Hub Diameter").Value = inputs[2] - 0.2
    m.Get("Rushton_1").Get("Blade Width").Value = inputs[3]
    m.Get("Rushton_1_1").Get("Blade Width").Value = inputs[3]
    
    # Export file
    m.Export(loc)
    
    # Save file with configuration of currently running M-Star model (to check thiings are working while optimisation is running)
    if os.path.isfile("running.msb"):
        os.system('rm running.msb')
    
    m.Save("running.msb")
 
    # Start simulation on two GPUs of provided ID
    compl = subprocess.run(['mpirun', '-np' , '2', solver_exe, '-i', 'input.xml', '-o', \
                            'out', '--gpu-ids', '0,1'], cwd=loc,capture_output=True)

    # If simulation has completed get data and process
    if compl.returncode == 0:
        
        #average kLa across probe locations
        df_p0 = pd.read_csv(loc+'/out/Stats/Probe_ProbeArray__0.txt', delimiter = "\t")
        df_p1 = pd.read_csv(loc+'/out/Stats/Probe_ProbeArray__1.txt', delimiter = "\t")
        df_p2 = pd.read_csv(loc+'/out/Stats/Probe_ProbeArray__2.txt', delimiter = "\t")
        df_p3 = pd.read_csv(loc+'/out/Stats/Probe_ProbeArray__3.txt', delimiter = "\t")
        df_p4 = pd.read_csv(loc+'/out/Stats/Probe_ProbeArray__4.txt', delimiter = "\t")
        df_p5 = pd.read_csv(loc+'/out/Stats/Probe_ProbeArray__5.txt', delimiter = "\t")
        t = df_p0['Time [s]'].values

        avgO2 = (df_p0['O2 [mol/L]'].values+df_p1['O2 [mol/L]'].values+df_p2['O2 [mol/L]'].values+\
            df_p3['O2 [mol/L]'].values+df_p4['O2 [mol/L]'].values+df_p5['O2 [mol/L]'].values)/6
        lnavgO2 = -np.log(1-avgO2/(0.032*0.0094))
        kLa_O2 = np.polyfit(t,lnavgO2,1)[0]

        avgCO2 = (df_p0['CO2 [mol/L]'].values+df_p1['CO2 [mol/L]'].values+df_p2['CO2 [mol/L]'].values+\
             df_p3['CO2 [mol/L]'].values+df_p4['CO2 [mol/L]'].values+df_p5['CO2 [mol/L]'].values)/6
        lnavgCO2 = np.log(4.464e-3/avgCO2)
        kLa_CO2 = np.polyfit(t,lnavgCO2,1)[0]

        # Get mixing time
        tauM = t[-1] - 10

        # Get maximum eddy dissipation rate
        df_maxeps = pd.read_csv(loc+'/out/Stats/GlobalVariables.txt', delimiter = "\t")
        maxEps = df_maxeps['e_t_avg_max'].values[-1]

        # Get time and space averaged eddy dissipation rate
        df_meaneps = pd.read_csv(loc+'/out/Stats/Fluid.txt', delimiter = "\t")
        meanEps = np.mean(df_meaneps['Mean Eddy Dissipation Rate [W/kg]'].values[-10:])

        # Get immersed gas volume 
        df_gasVol = pd.read_csv(loc+'/out/Stats/Particles_Bubbles.txt', delimiter = "\t")
        gasVol = np.mean(df_gasVol['Total Volume [m^3]'].values[-10:])

        # Return performance characteristics
        return((kLa_O2,kLa_CO2,tauM,maxEps,meanEps,gasVol*1000))

    # If simulation did not complete report error and return zero performance
    else: 
        print("MStar Run Error: ", compl.returncode, compl.stderr, compl.stdout)
        os.system('rm -r '+loc)
        return((0,0,0))
    
# Function that takes parameters from Ax, calls the function run the M-Star model and processes the results to provide yield prediction    
def TEACFD(parameterization):
    """
    Function that sist as intermediary between Ax and function that sets up, runs and processes M-Star model to provide performance characteristics

    Parameters
    ----------
    parameterization : List of Dictionaries from Ax

    Returns
    -------
    Dictionary with predicted yield in Ax format
    """
    # Local variables for input parameters
    rpm=parameterization['RPM']
    Q=parameterization['Gas Flow Rate']
    d=parameterization['Impeller Diameter']
    w=parameterization['Blade Width']

    # Get CFD predicted performance characteristics
    cfdResults = getBioreactorMetrics((rpm,Q*60000,d,w))
    
    # Set bioreactor characteristics
    workingVolume = 20000 #workingVolume = 41600 for 42kl case
    tankDiameter = 2.34 #tankDiameter = 3.0 for 42kl case
    impellerDiameter = tankDiameter/3
    backPressure = 1.3
    superficialVel = Q/(math.pi*(tankDiameter/2)**2)
    moleFracO2 = 0.21
    moleFracCO2 = 0.03
    initVol = 0.8 * workingVolume
    initCells = 4e6
    temp = 310
    powerNumber = 5
    mediumDensity = 1000
    mediumViscosity = 9e-4
    vesselVolDay = 0.0
    perfLactateRate = 5.0
    perfAmmrate = 5.0

    # Create bioreactor class
    bioreactor = biomodels.Bioreactor(wv=workingVolume,t=tankDiameter,d=impellerDiameter,n=rpm,p_back=backPressure,\
                                   u_s=superficialVel,mf_O2_gas=moleFracO2,mf_CO2_gas=moleFracCO2,v0=initVol,\
                                    ns=initCells,Temp=temp,Np=powerNumber,rho=mediumDensity,mu=mediumViscosity,\
                                        vvd=vesselVolDay,perfAMM=perfAmmrate,perfLAC=perfLactateRate)
    # Set cell characteristics
    growthRate = 0.029
    glutamineUptakeRate = 0
    glucoseUptakeRate = 0
    oxygenUptakeRate = 0.48963
    uptakeList = [glutamineUptakeRate, glucoseUptakeRate, oxygenUptakeRate]
    carbonDioxideProdRate = 0.593197
    ammoniaProductionRate = 0.013571
    lactateProductionRate = 0.135707
    prodList = [carbonDioxideProdRate, ammoniaProductionRate, lactateProductionRate]
    massDensity = 1030
    cellRadius = 18e-6
    wetmass = 3000
    dryMassFraction = 0.3
    ammoniaLimit = 5
    lactateLimit = 50
    CO2Limit = 100
    turbLengthLimit = 20e-6
    limitsList = [ammoniaLimit, lactateLimit, CO2Limit, turbLengthLimit]

    # Create cell class
    cell = biomodels.Cell(mu=growthRate,uptakes=uptakeList,prod=prodList,rho=massDensity,rad=cellRadius,\
                       wetmass=wetmass,dmf=dryMassFraction,limits=limitsList)
    
    # Constrain performance to mixing time and hydrodynamic stress constraints
    if (cfdResults[2]>1/cfdResults[0]) or (cfdResults[2]>1/cfdResults[1]) or (cfdResults[3] > 100):
        result = {'yield':(bioreactor.ns*1000*cell.wetmass*1e-12,None)}
        return (result)

    # Run SciPy's odeInt to solved coupled differential equations to get O2 and CO2 constrained yield
    dummy,a = biomodels.bioreactorODEs(bioreactor,cell,200,klaInO2=float(cfdResults[0]),\
                                    klaInCO2=float(cfdResults[1]),C_CO2_in=1.66e-5,C_O2_in=0.0087,
                                    Q=Q*3600000,V=float(cfdResults[5]))

    # Create dictionary for Ax
    result = {'yield':(min(a['O2'],a['CO2']),None)}

    # Return dictionary
    return (result)

#   Ax set-up and execution
best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "RPM", 

            
            "type": "range", 
            "bounds": [20.0, 120.0],
            #"bounds": [40.0, 160.0],

            "value_type": "float",
        },
        {
            "name": "Gas Flow Rate",  
            "type": "range", 
            "bounds": [3.33e-3, 3.33e-2],
            #"bounds": [6.93e-3, 6.93e-2],
            "value_type": "float", 
        },
        {
            "name": "Impeller Diameter", 
            "type": "range", 
            "bounds": [0.7, 1.2],
            #"bounds": [0.9, 1.5],
            "value_type": "float",
        },
        {
            "name": "Blade Width", 
            "type": "range", 
            "bounds": [0.1, 0.3],
            #"bounds": [0.15, 0.4],
            "value_type": "float",
        },
    ],
    experiment_name="test",
    evaluation_function=TEACFD,
    #outcome_constraints=["CO2 >= -10.0","Mixing >= -10.0"],  
    objective_name="yield",
    total_trials=30,
)