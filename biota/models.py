"""
This module provides performance modelling and yield prediction funcionality
primarily for cultivated meat technical analaysis, but can easily be adapted 
for other cellular agriculture applications

Usage: import biota.models
"""

import numpy as np
import pandas as pd

from scipy.integrate import odeint

import matplotlib.pyplot as plt
import math

## PHYSICAL PROPERTIES ##

#Henry's coefficients

H_O2 = 1.3 * math.exp(-1700/8.314*(1/310-1/298.15))     #[mmol/(L bar)]
H_CO2 = 34 * math.exp(-2400/8.314*(1/310-1/298.15))     #[mmol/(L bar)]



#molar mass of oxygen (O2)
mw_O2 = 32      #[g/mol]
mw_CO2 = 44.01  #[g/mol]

#water vapour pressure
H2O_vapP = 0.06 #[bar]

#########################

## CLASSES ##

class Bioreactor:
    
    """
    Basic class to represent a stirred aerated tank bioreactor. Everything but the cell goes in here.
    
    Attributes
    ----------
    wv : float
        working volume of the bioreactor, in litres
    t : float
        Bioreactor diameter, assumes circular cross section, metres
    d : float
        Impeller diameter, in metres
    n : float
        Impeller RPM
    p_back : float
        Back pressure, in [bar]
    u_s : float
        superficial velocity of sparger, in metres per second
    mf_O2_gas : float
        Mole fraction of oxygen in sparged gas, no dimensions
    v0 : float
        Initial volume of media and cells, in litres
    ns : float
        Initial number of cells, per milli-litre
    Temp : float
        Operating temperature, in Kelvin
    Np : float
        Impeller power number, no dimensions
    rho : float
        Density of the medium, kg/m3
    mu : float
        Dynamic viscosity of the medium, in Pascal seconds
    vvd : float
        Vessel volumes per day perfusion rate, per day
    perfAMM : float
        Ammonia concentration perfusion removal rate, in milli-moles per litres
    perfLAC : float
        Lactate concentration perfusion removal rate, in milli-moles per litres
    tipSpeed : float
        Impeller tip speed, in metres per second
    power : float
        Impeller power, in Watts
    tau : float
        Mixing time, in seconds
    
    Methods
    -------
    calctipSpeed()
        Calculates the tip speed of the impeller
    calcPower()
        Calculates the impeller power from the power number, media density, impeller diameter and rpm
    calctau()
        Calculates mixing time estimate based on power, working volume, dimensions and media density
    
    
    """
    
    
    def __init__(self,wv,t,d,n,p_back,u_s,mf_O2_gas,mf_CO2_gas,v0,ns,Temp,Np,rho,mu,vvd,perfAMM,perfLAC):
        """
        Parameters
        ----------
        wv : float        
            working volume of the bioreactor, in litres
        t : float
            Bioreactor diameter, assumes circular cross section, metres
        d : float
            Impeller diameter, in metres
        n : float
            Impeller RPM
        p_back : float
            Back pressure, in [bar]
        u_s : float
            superficial velocity of sparger, in metres per second
        mf_O2_gas : float
            Mole fraction of oxygen in sparged gas, no dimensions
        v0 : float
            Initial volume of media and cells, in litres
        ns : float
            Initial number of cells, per milli-litre
        Temp : float
            Operating temperature, in Kelvin
        Np : float
            Impeller power number, no dimensions
        rho : float
            Density of the medium, kg/m3
        mu : float
            Dynamic viscosity of the medium, in Pascal seconds
        vvd : float
            Vessel volumes per day perfusion rate, per day
        perfAMM : float
            Ammonia concentration perfusion removal rate, in milli-moles per litres
        perfLAC : float
            Lactate concentration perfusion removal rate, in milli-moles per litres
        tipSpeed : float
            Impeller tip speed, in metres per second
        power : float
            Impeller power, in Watts
        tau : float
            Mixing time, in seconds
            
        """
        
        self.wv = wv                    #working volume [litres]
        self.t = t                      #tank diameter [m]
        self.d = d                      #impeller diameter [m]
        self.n = n                      #impeller rpm [rpm]
        self.p_back = p_back            #back pressure [bar]
        self.u_s = u_s                  #gas superficial velocity (gas volume flow rate / tank X sect area) [m/s]
        self.mf_O2_gas = mf_O2_gas      #mole fraction of oxygen in sparged gas[-]
        self.mf_CO2_gas = mf_CO2_gas    #mole fraction of oxygen in sparged gas[-]
        self.v0 = v0                    #initial volume of media and cells [litres]
        self.ns = ns                    #starting number of cells [1/mlitre]
        self.Temp = Temp                #operating temp [K]
        self.Np = Np                    #impeller power number [-]
        self.rho = rho                  #medium density [kg/m3]
        self.mu = mu                    #medium dynamic viscosity [Pa.s]
        self.vvd = vvd                  #vessel volumes per day perfusion rate [1/day]        
        self.perfAMM = perfAMM          #concentration removal of ammonia through perfusion [mmol/L] 
        self.perfLAC = perfLAC          #concentration removal of lactate through perfusion [mmol/L]
        
        self.tipSpeed = 2*math.pi*(self.d/2)*self.n/60              

        self.power = 2*self.Np*(self.n/60)**3*self.rho*self.d**5    

        self.tau = 6 * self.t**(2/3) * \
            (self.power/(self.wv/1000*self.rho))**(-1/3) * \
            (self.d/self.t)**(-1/3) * \
            (self.wv/1000/(math.pi*(self.t/2)**2)/self.t)**2.5

    def calctipSpeed(self):
        """Calculates the impeller tip speed.
        """
        
        self.tipSpeed = 2*math.pi*(self.d/2)*self.n/60

    def calcpower(self):
        """Calculates the impeller power.
        """
        
        self.power = 2*self.Np*(self.n/60)**3*self.rho*self.d**5

    def calctau(self):
        """Calculates the mixing time.
        """
        
        self.tau = 6 * self.t**(2/3) * \
        (self.power/(self.wv/1000*self.rho))**(-1/3) * \
        (self.d/self.t)**(-1/3) * \
        (self.wv/1000/(math.pi*(self.t/2)**2)/self.t)**2.5
        
class Cell:
    """
    Basic class to represent the cell characteristics
    
    Attributes
    ----------
    mu : float
        Growth rate, per hour
    UPglut : float
        Glutamine uptake rate, in milli-moles per gram dry weight per hour
    UPgluc : float
        Glucose uptake rate, in milli-moles per gram dry weight per hour
    UPO2 : float
        Oxygen uptake rate, in milli-moles per gram dry weight per hour
    PRODCO2 : float
        Carbon dioxide production rate, in milli-moles per gram dry weight per hour
    PRODCO2 : float
        Carbon dioxide production rate, in milli-moles per gram dry weight per hour
    PRODamm : float
        Ammonia production rate, in milli-moles per gram dry weight per hour
    PRODlac : float
        Lactate production rate, in milli-moles per gram dry weight per hour
    rho : float
        Cell mass denisty, in kg/m3
    rad : float
        Cell radius, assuming spherical, in metres
    wetmass : float
        Cell wetmass, in pico-grammes
    dmf : float
        Cell dry mass fraction, no dimensions
    limits : array/list
        Limits for cell growth for [ammonia(milli-moles/litre), lactate(milli-moles/litre), CO2(millibar), energy dissipation length(metres)]
    """
    
    def __init__(self,mu,uptakes,prod,rho,rad,wetmass,dmf,limits):
        """
        Parameters
        ----------
        mu : float
            Growth rate, per hour
        UPglut : float
            Glutamine uptake rate, in milli-moles per gram dry weight per hour
        UPgluc : float
            Glucose uptake rate, in milli-moles per gram dry weight per hour
        UPO2 : float
            Oxygen uptake rate, in milli-moles per gram dry weight per hour
        PRODCO2 : float
            Carbon dioxide production rate, in milli-moles per gram dry weight per hour
        PRODCO2 : float
            Carbon dioxide production rate, in milli-moles per gram dry weight per hour
        PRODamm : float
            Ammonia production rate, in milli-moles per gram dry weight per hour
        PRODlac : float
            Lactate production rate, in milli-moles per gram dry weight per hour
        rho : float
            Cell mass denisty, in kg/m3
        rad : float
            Cell radius, assuming spherical, in metres
        wetmass : float
            Cell wetmass, in pico-grammes
        dmf : float
            Cell dry mass fraction, no dimensions
        limits : array/list
            Limits for cell growth for [ammonia(milli-moles/litre), lactate(milli-moles/litre), CO2(millibar), energy dissipation length(metres)]
        """
        
        self.mu = mu            #growth rate [1/hr]
        self.UPglut = uptakes[0]    #glutamine uptake rate [mmol / gDW hr]
        self.UPgluc = uptakes[1]    #glucose uptake rate [mmol / gDW hr]
        self.UPO2 = uptakes[2]        #oxygen uptake rate [mmol / gDW hr]
        self.PRODCO2 = prod[0]  #co2 uptake rate [mmol / gDW hr]
        self.PRODamm = prod[1]  #ammonia uptake rate [mmol / gDW hr]
        self.PRODlac = prod[2]  #lactate uptake rate [mmol / gDW hr]
        self.rho = rho          #mass density [kg/m3]
        self.rad = rad          #cell radius [m]
        self.wetmass = wetmass  #wet mass [pg]
        self.dmf = dmf          #dry mass fraction [-]
        self.limits = limits    #amm,lac,co2,stress growth constraints
    
    def doubleTime(self):
        """Calculates cell doubling time
        """
        
        return math.log(2)/self.mu
    
    def cellVolume(self):
        """Calculates cell volume
        """
        
        return 4/3*math.pi*(self.rad)**3 #[m3]
    
#############    
    

## FUNCTIONS ##

# Yield prediction method using coupled ODE's
def bioreactorODEs(b,c,duration,klaInO2,klaInCO2):
    """
    Predicts yield based on dissolved oxygen and carbon dioxide concentrations which are solved over time
    through the use of SciPy's odeInt function to solve coupled ordinary differential equations for both species
    
    Parameters
    ----------
    b : Class
        Bioreactor class instance
    c : Class
        Cell class instance
    klaInO2 : float
        Oxygen mass transfer coefficient
    klaInCO2 : float
        Carbon dioxide mass transfer coefficient

    Returns
    -------
    dataframe : dataframe
        Time history of all performance and constraint values.
    limitsDict : dictionary
        Yield limits by constraint
    
    """

    #array length
    L = 10000

    #make time array for duration [hrs]
    t=np.linspace(0,duration,L) #[hrs]

    #evaluate number of cells over time
    N0=b.ns*b.v0*1000 #[-]
    N=N0*np.exp(c.mu*t) #[-]

    #grams dry weight of cells with time
    gDW=N*c.wetmass*1e-12*c.dmf #[grams]
    
    #functions for change in dissolved gas concentrations
    def do2dt(y,t): #Oxygen
        (Cg,C) = y
        dCgdt = -(H_O2d*Cg - C)*klaInO2*3600 + Q*(C_O2_in - Cg)/V        
        dCdt = klaInO2*(H_O2d*Cg - C)*3600 - c.UPO2*b.ns*math.exp(c.mu*t)*c.wetmass*c.dmf*1e-12
        return [dCgdt,dCdt]

    def dco2dt(y,t): #Carbon dioxide
        (Cg,C) = y
        dCgdt = -(H_CO2d*Cg - C)*klaInCO2*3600 + Q*(C_CO2_in - Cg)/V
        dCdt = klaInCO2*(H_CO2d*Cg - C)*3600 + c.PRODCO2*b.ns*math.exp(c.mu*t)*c.wetmass*c.dmf*1e-12
        return [dCgdt,dCdt]
    
    #dolve coupled ODEs for oxygen and carbon dioxide concentrations
    solnO2 = odeint(do2dt, [0.0087,1e-6], t)
    solnCO2 = odeint(dco2dt,[1.66e-5,1e-6] , t)

    #Extract concentrations with time and plot. Used for testing, commented out for production use.
    # concO2g = solnO2[:,0]
    # concO2l = solnO2[:,1]
    # concCO2g = solnCO2[:,0]
    # concCO2l = solnCO2[:,1]
    
    # plt.figure(figsize=(10,6))
    # plt.plot(t,concO2l)
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Time [hrs]')
    # plt.ylabel('Liquid O2 [mol/L]')
    # plt.show()

    # plt.figure(figsize=(10,6))
    # plt.plot(t,concO2g)
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Time [hrs]')
    # plt.ylabel('Gas O2 [mol/L]')
    # plt.show()

    # plt.figure(figsize=(10,6))
    # plt.plot(t,concCO2l)
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Time [hrs]')
    # plt.ylabel('Liquid CO2 [mol/L]')
    # plt.show()

    # plt.figure(figsize=(10,6))
    # plt.plot(t,concCO2g)
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Time [hrs]')
    # plt.ylabel('Gas CO2 [mol/L]')
    # plt.show()


    #find constraint limits
    indexCO2 = min(concCO2l[concCO2l<0.00348].shape[0],concCO2l.shape[0]-1)
    indexO2 = min(concO2l[concO2l>0].shape[0],concO2l.shape[0]-1)

    #report yield for each constraint limit and make dictionary to return
    limitCO2 = gDW[indexCO2]/c.dmf/b.v0
    limitO2 = gDW[indexO2]/c.dmf/b.v0

    limitsDict = {"CO2":limitCO2,"O2":limitO2}  

    #constract dataframe with relevant constraints and parameters over time
    dataframe = pd.DataFrame({'Time [hr]': t, 'CO2 [mol/L]': concCO2l.flatten(), 'O2 [mol/L]': concO2l.flatten()})

    return (dataframe,limitsDict)

# Yield prediction function using prior method    
def yieldModel(b,c,duration,klaInO2,tauMIn,epsIn):
    """
    Production performance function to predict overall yield and yield limits by constraint type.
    Uses array broadcasting to evaluate time dependant behaviour in a single step, based on assumption of steady fluxes.
    Models and assumptions are as described in the publication of Humbird https://doi.org/10.31224/osf.io/795su 

    Parameters
    ----------
    b : Class
        Bioreactor class instance
    c : Class
        Cell class instance
    duration : int
        Duration for which to assess performance, in hours.
    klaIn : (float|function|boolean)
        Input for volumetric mass transfer coefficient kla which may be:
        A float, for use with single data point, e.g. from CFD
        A function, for use with apriori determined relationship to rpm and u_s , e.g. from CFD or lab characterisation
        A boolean (must be == False), to use literature model
    tauMIn : (float|function|boolean)
        Input for mixing time tauM which may be:
        A float, for use with single data point, e.g. from CFD
        A function, for use with apriori determined relationship to rpm and u_s , e.g. from CFD or lab characterisation
        A boolean (must be == False), to use literature model
    epsIn : (float|function|boolean)
        Input for maximum eddy dissipiation rate epsMax which may be:
        A float, for use with single data point, e.g. from CFD
        A function, for use with apriori determined relationship to rpm and u_s , e.g. from CFD or lab characterisation
        A boolean (must be == False), to use literature model

    Returns
    -------
    dataframe : dataframe
        Time history of all performance and constraint values.
    limitsDict : dictionary
        Yield limits by constraint
    """
    
    #array length
    L = 10000

    #make time array for duration [hrs]
    t=np.linspace(0,duration,L) #[hrs]

    #evaluate number of cells over time
    N0=b.ns*b.v0*1000 #[-]
    N=N0*np.exp(c.mu*t) #[-]

    #grams dry weight of cells with time
    gDW=N*c.wetmass*1e-12*c.dmf #[grams]

    #vol of cells with time
    vol_cells = N * c.cellVolume() * 1000 #[litres]

    #volume of media and cells with time, assuming increase is due to cell growth only
    V_t = b.v0  #[litres]

    #volFrac = vol_cells / (V_t + vol_cells)
    volFrac = N/1e6/(V_t*1000)*c.wetmass/1.03/1e6

    #media height
    mH = (V_t/1000)/(math.pi*(b.t/2)**2) #[metres]

    #pressure at bottom [bar]
    p_bottom = (b.rho * 9.81 * mH)/1e5 + b.p_back



    #oxygen uptake and co2 production rates 
    O2_UR  = c.UPO2*gDW*1e-6 #[kmol/hr]
    CO2_PR = c.PRODCO2*gDW*1e-6  #[kmol/hr]

    #concentrations of co2, ammonia and lactate
    AMM=c.PRODamm/c.mu*N0*(np.exp(c.mu*t)-1)*c.wetmass*1e-12*c.dmf/V_t - t*b.vvd*b.wv*b.perfAMM/24 #[mmol/L]
    LAC=c.PRODlac/c.mu*N0*(np.exp(c.mu*t)-1)*c.wetmass*1e-12*c.dmf/V_t - t*b.vvd*b.wv*b.perfLAC/24 #[mmol/L]

    #gas in flowrate in [kmol/hr]
    gas_in = 3600*(b.u_s*math.pi*(b.t/2)**2*p_bottom*1e5)  / (8.314 * b.Temp) / 1000 #[kmol/hr]

    #rate of gaseous output into headspace
    N2_out = gas_in * (1 - b.mf_O2_gas)   #[kmol/hr]
    O2_out = gas_in * b.mf_O2_gas - O2_UR #[kmol/hr]
    CO2_out = CO2_PR                    #[kmol/hr] 
    H2O_out = H2O_vapP/b.p_back*(N2_out+O2_out+CO2_out)/(1-H2O_vapP/b.p_back) #[kmol/hr]

    #total gas out rate into headspace in m3/s
    gas_outa = ((N2_out+O2_out+CO2_out+H2O_out)*1000)*8.314*b.Temp/(b.p_back*1e5)/3600
    gas_out = ((N2_out+O2_out+CO2_out+H2O_out))*0.08314*b.Temp/(b.p_back)/3600
    u_s_top = gas_out/(math.pi*(b.t/2)**2)

    #mole fraction of oxgyen and co2 in outlet gas - check this is right, not fraction at top of liquid, given subsequent pCO2 calc
    mf_O2_out = O2_out/(N2_out+O2_out+CO2_out+H2O_out)
    mf_CO2_out = CO2_out/(N2_out+O2_out+CO2_out+H2O_out)

    #co2 liquid concentration, in pCO2
    pCO2 = mf_CO2_out * b.p_back * 1000 #[mbar]

    #o2 Delta C log mean difference 
    CsatO2_t = b.p_back * mf_O2_out * H_O2 * mw_O2
    CsatO2_b = p_bottom * b.mf_O2_gas * H_O2 * mw_O2

    DO_t = b.mf_O2_gas * H_O2 * mw_O2 * 0.2    
    DO_b = DO_t * p_bottom / b.p_back

    diff_t = (CsatO2_t-DO_t)[(CsatO2_t-DO_t)>0]
    if len(diff_t)==0:
        limitsDict = {"CO2":0,"O2":0,"Mixing":0,\
                  "Superficial Velocity Top":0,"Stress":0,\
                    "Ammonia":0,"Lactate":0}  
        
        return (0,limitsDict)


    #t = t[:diff_t.shape[0]] # hard oxygen transfer rate limit
    #print(diff_t)
    CsatO2_t = CsatO2_t[:diff_t.shape[0]]

    DeltaC_O2_lmd = ((CsatO2_b - DO_b) - (CsatO2_t - DO_t))/np.log((CsatO2_b - DO_b)/(CsatO2_t - DO_t))/mw_O2
    DeltaC_O2_lmd = DeltaC_O2_lmd[:diff_t.shape[0]]
    
    #limit all arrays by hard oxygen transfer time
    O2_UR = O2_UR[:diff_t.shape[0]]
    CO2_PR = CO2_PR[:diff_t.shape[0]]
    pCO2 = pCO2[:diff_t.shape[0]]
    u_s_top = u_s_top[:diff_t.shape[0]]
    gas_out = gas_out[:diff_t.shape[0]]
    gas_outa = gas_outa[:diff_t.shape[0]]
    volFrac = volFrac[:diff_t.shape[0]]
    mf_O2_out = mf_O2_out[:diff_t.shape[0]]
    gDW = gDW[:diff_t.shape[0]]
    AMM = AMM[:diff_t.shape[0]]
    LAC = LAC[:diff_t.shape[0]]

    # perf criteria
    #mixing time
    if tauMIn == False:
        tau_mix = 6 * b.t**(2/3) * \
            (b.power/(V_t/1000*b.rho))**(-1/3) * \
            (b.d/b.t)**(-1/3) * \
            (mH/b.t)**2.5
    elif type(tauMIn) == float: 
        tau_mix = tauMIn
    elif callable(tauMIn):
        tau_mix = tauMIn()
    else: 
        print("Error in tauMIn")
        exit()
    
    #theoretical kLa O2
    if klaInO2 == False:
        kla_O2_theory = 0.075 * (b.power/(V_t/1000))**0.47 * ((b.u_s+u_s_top)/2)**0.8
    elif type(klaInO2) == float: 
        kla_O2_theory =  klaInO2
    elif callable(klaInO2):
        kla_O2_theory = np.ones(L) * klaInO2()
    else: 
        print("Error in klaIn")
        exit()


    #Kolmogorov eddy disspitaion length scale from dissipated energy
    if epsIn == False:
        eps = b.power/(V_t/1000)/b.rho
        epsMax = 50*eps
        #epsMax = 25*eps
        lambdaK = ((b.mu/b.rho)**3/epsMax)**0.25
    elif type(epsIn) == float:
        lambdaK = ((b.mu/b.rho)**3/epsIn)**0.25
    elif callable(epsIn):
        lambdaK = ((b.mu/b.rho)**3/epsIn())**0.25
    else: 
        print("Error in epsIn")
        exit()
        

    #required kla to meet oxygen uptake rate
    kla_O2_needed = O2_UR / (V_t*(1-volFrac)) * 1e6 / DeltaC_O2_lmd / 3600
    kla_O2_Ratio = kla_O2_needed/kla_O2_theory
   
    #required mixing time
    mixing_O2 = tau_mix*kla_O2_needed

    #find constraint limits
    indexpCO2 = min(pCO2[pCO2<c.limits[2]].shape[0],pCO2.shape[-1]-1)
    indexkla_O2 = min(kla_O2_Ratio[kla_O2_Ratio<1].shape[0],kla_O2_Ratio.shape[-1]-1)
    indexmixing_O2 = min(mixing_O2[mixing_O2<1].shape[0],mixing_O2.shape[-1]-1)
    indexustop = min(u_s_top[u_s_top<=0.006].shape[0],u_s_top.shape[-1]-1)

    indexAMM = min(AMM[AMM<c.limits[0]].shape[0],AMM.shape[-1]-1)
    indexLAC = min(LAC[LAC<c.limits[1]].shape[0],LAC.shape[-1]-1) 

    #report time and yield for each constraint limit
    limitpCO2 = gDW[indexpCO2]/c.dmf/V_t
    limitkla_O2 = gDW[indexkla_O2]/c.dmf/V_t
    limitmixing_O2 = gDW[indexmixing_O2]/c.dmf/V_t
    limitustop = gDW[indexustop]/c.dmf/V_t
    limitAMM = gDW[indexAMM]/c.dmf/V_t
    limitLAC = gDW[indexLAC]/c.dmf/V_t
    
    if lambdaK < c.limits[3]:
        limitlambdaK = 0
    else:
        limitlambdaK = gDW[-1]/c.dmf/V_t

    limitsDict = {"CO2":limitpCO2,"O2":limitkla_O2,"Mixing":limitmixing_O2,\
                  "Superficial Velocity Top":limitustop,"Stress":limitlambdaK,\
                    "Ammonia":limitAMM,"Lactate":limitLAC}  
        
    #constract dataframe with relevant constraints and parameters over time
    dataframe = pd.DataFrame({'Time [hr]': t[:diff_t.shape[0]], 'Mixing Time [s]': tau_mix, 'Required/Theoretical kLa [-]': kla_O2_Ratio \
                             , 'Required kLa * Mixing Time [-]': kla_O2_needed*tau_mix, 'pCO2 [mbar]': pCO2, 'Ammonia [mmol/L]': AMM \
                             , 'Lactate [mmol/L]': LAC, 'Superficial Gas Top [m/s]': u_s_top, 'Cell Density [wet g/L]': gDW/c.dmf/V_t})

    return (dataframe,limitsDict)


def brute(count,b,c,dbls,rpmlims,uslims,nslims,graphs):
    """
    Brute force parameter sweep on impeller rpm, superficial sparge velcoity u_s and initial cell number n_s. 
    Works reasonably well for counts <=30 after which the computation time increases.
    Care shoudl be taken with the initial cell number range, as high values that don;t grow may still report a high yield.
    Further work needed to assess Jumba or similar modules to parallelize in future.

    Parameters
    ----------
    count : integer
        The number of values to consider for rpm, u_s and n_s.
    b : Bioreactor class instance
        Bioreactor class instance.
    c : Cell class instance
        Cell class instance.
    dbls : float
        Number of cell doublings to consider.
    rpmlims : tuple of floats
        Upper and lower limits of the rpm range.
    uslims : tuple of floats
        Upper and lower limits of the u_s range.
    nslims : tuple of floats
        Upper and lower limits of the ns range.
    graphs : Boolean
        Whether to generate graphs or not.

    Returns
    -------
    dataframe
        Dataframe of the overall and per constraint maximum yield.

    """

    #calc times from doublings
    time1 = dbls * math.log(2)/c.mu
    
    #sweep rpm, superficial gas velocity and starting cell density
    
    #make arrays for each
    rpms = np.linspace(rpmlims[0],rpmlims[1],count)
    supers = np.linspace(uslims[0],uslims[1],count)
    #dens = np.linspace(nslims[0],nslims[1],count)
    
    #make 3D and 2D mesh grids to store date
    rr, ss = np.meshgrid(rpms, supers,  indexing='ij')
    rm, sm = np.meshgrid(rpms, supers, indexing='ij')

    
    #yields over all rpm/u_s/n_s combinations - initialise to zero valued meshgrids
    ammoniaYield = rm*0
    lactateYield = rm*0
    pCO2Yield = rm*0
    klaYield = rm*0
    mixingYield = rm*0
    lambdaKYield = rm*0
    ustopYield = rm*0
    overallYield = rm*0
    
    #loop through each combination of rpm/u_s/n_s
    for i in range(count):
        for j in range(count):
            # treat xv[i,j], yv[i,j]
            b.n = rr[i,j]
            b.u_s = ss[i,j]
            #b.ns = dd[i,j,k]
            b.calctipSpeed()
            b.calcpower()
            b.calctau()
            dummy,aa = yieldModel(b,c,time1,False,False,False)

            #determine specific yields
            ammoniaYield[i,j] = min(aa['Ammonia'],aa['Stress'],aa['Superficial Velocity Top'])
            lactateYield[i,j] = min(aa['Lactate'],aa['Stress'],aa['Superficial Velocity Top'])
            pCO2Yield[i,j] = min(aa['CO2'],aa['Stress'],aa['Superficial Velocity Top'])
            klaYield[i,j] = min(aa['O2'],aa['Stress'],aa['Superficial Velocity Top'])
            mixingYield[i,j] = min(aa['Mixing'],aa['Stress'],aa['Superficial Velocity Top'])
            lambdaKYield[i,j] = aa['Stress']
            ustopYield[i,j] = aa['Superficial Velocity Top']
            
            #determine overall yield and duration
            overallYield[i,j] = min(ammoniaYield[i,j] ,lactateYield[i,j],pCO2Yield[i,j],klaYield[i,j],\
                                    mixingYield[i,j],lambdaKYield[i,j],ustopYield[i,j])
                
                
    
    if graphs:
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,3,1)
        plt.contourf(rm, sm, overallYield)
        plt.colorbar()
        plt.title('Overall Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.subplot(2,3,2)
        plt.contourf(rm, sm, ammoniaYield)
        plt.colorbar()
        plt.title('Ammonia Concentration\n Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")
        plt.tight_layout()

        plt.subplot(2,3,3)
        plt.contourf(rm, sm, lactateYield)
        plt.colorbar()
        plt.title('Lactate Concentration\n Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.subplot(2,3,4)
        plt.contourf(rm, sm, klaYield)
        plt.colorbar()
        plt.title('Oxygen Transfer\n Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")
        
        plt.subplot(2,3,5)
        plt.contourf(rm, sm, pCO2Yield)
        plt.colorbar()
        plt.title('Carbon Dioxide Concentration \nLimited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.subplot(2,3,6)
        plt.contourf(rm, sm, mixingYield)
        plt.colorbar()
        plt.title('Mixing Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.tight_layout()
        plt.show()

    #construct dataframe to provide overall and per constraint maximum yields
    dataframe = pd.DataFrame({'Constraint': ['overall','ammonia','lactate','CO2','kla','mixing','hydrodynamic stress','superficial velocity'],\
                              'Maximum Yield [g/L wet]':[overallYield.max(),ammoniaYield.max(),lactateYield.max(),pCO2Yield.max(),klaYield.max(),\
                                mixingYield.max(),lambdaKYield.max(),ustopYield.max()]})
    
    return (dataframe)
    
###############