 # 
 # This file is part of the biotechan distribution (https://github.com/xxxx or http://xxx.github.io).
 # Copyright (c) 2023 Upstream Applied Science Ltd.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #
 
"""
This module provides performance modelling and yield prediction funcionality
primarily for cultivated meat technical analaysis, but can easily be adapted 
for other cellular agriculture applications

Usage: import biotechan
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import math

## PHYSICAL PROPERTIES ##

#Henry's coefficient for oxygen 
H_O2 = 1.04 #[mmol/(L bar)]

#molar mass of oxygen (O2)
mw_O2 = 32   #[g/mol]

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
    
    
    def __init__(self,wv,t,d,n,p_back,u_s,mf_O2_gas,v0,ns,Temp,Np,rho,mu,vvd,perfAMM,perfLAC):
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
        
        self.wv = wv         #working volume [litres]
        self.t = t           #tank diameter [m]
        self.d = d           #impeller diameter [m]
        self.n = n           #impeller rpm [rpm]
        self.p_back = p_back #back pressure [bar]
        self.u_s = u_s       #gas superficial velocity (gas volume flow rate / tank X sect area) [m/s]
        self.mf_O2_gas = 0.2 #mole fraction of oxygen in sparged gas[-]
        self.v0 = v0         #initial volume of media and cells [litres]
        self.ns = ns         #starting number of cells [1/mlitre]
        self.Temp = Temp     #operating temp [K]
        self.Np = Np         #impeller power number [-]
        self.rho = rho       #medium density [kg/m3]
        self.mu = mu         #medium dynamic viscosity [Pa.s]
        self.vvd = vvd       #vessel volumes per day perfusion rate [1/day]        
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

#product performance evaluation function    
def prodPerf(b,c,duration):
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

    Returns
    -------
    dataframe : dataframe
        Time history of all performance and constraint values.
    limitsDescriptList : list of strings
        Names of constraints index linked to limitsList
    limitsList : list of tuples
        Time and cell wet mass yield limits for each constraint
    """
    
    #make time array for duration [hrs]
    t=np.linspace(0,duration,1000) #[hrs]
    
    #evaluate number of cells over time
    N0=b.ns*b.v0*1000 #[-]
    N=N0*np.exp(c.mu*t) #[-]
    
    #grams dry weight of cells with time
    gDW=N*c.wetmass*1e-12*c.dmf #[grams]
    
    #vol of cells with time
    vol_cells = N * c.cellVolume() * 1000 #[litres]

    #volume of media and cells with time, assuming increase is due to cell growth only
    V_t = b.v0 + vol_cells - vol_cells[0] #[litres]
    volFrac = vol_cells/(V_t)
    
    #media height
    mH = (V_t/1000)/(math.pi*(b.t/2)**2) #[metres]
    
    #pressure at bottom [bar]
    p_bottom = (b.rho * 9.81 * mH)/1e5 + b.p_back
    
    #mixing time
    tau_mix = 6 * b.t**(2/3) * \
            (b.power/(V_t/1000*b.rho))**(-1/3) * \
            (b.d/b.t)**(-1/3) * \
            (mH/b.t)**2.5
    
    #theoretical kLa
    kla_theory = 0.075 * (b.power/(V_t/1000))**0.47 * b.u_s**0.8
    
    #oxygen uptake and co2 production rates 
    O2_UR  = c.UPO2*gDW*1e-6 #[kmol/hr]
    CO2_PR = c.PRODCO2*gDW*1e-6  #[kmol/hr]

    #concentrations of co2, ammonia and lactate
    CO2=c.PRODCO2/c.mu*N0*(np.exp(c.mu*t)-1)*c.wetmass*1e-12*c.dmf/V_t #[mmol/L]
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
    gas_out = ((N2_out+O2_out+CO2_out+H2O_out)*1000)*8.314*b.Temp/(b.p_back*1e5)/3600
    u_s_top = gas_out/(math.pi*(b.t/2)**2)
    
    
    #mole fraction of oxgyen and co2 in outlet gas - check this is right, not fraction at top of liquid, given subsequent pCO2 calc
    mf_O2_out = O2_out/(N2_out+O2_out+CO2_out+H2O_out)
    mf_CO2_out = CO2_out/(N2_out+O2_out+CO2_out+H2O_out)

    #co2 liquid concentration, in pCO2
    pCO2 = mf_CO2_out * b.p_back * 1000 #[mbar]

    Csat_t = b.p_back * mf_O2_out * H_O2 * mw_O2
    Csat_b = p_bottom * b.mf_O2_gas * H_O2 * mw_O2

    DO_t = b.mf_O2_gas * H_O2 * mw_O2 * 0.2    #check physics on this one
    DO_b = DO_t * p_bottom / b.p_back
    
    #log mean squared concentration difference
    if ((Csat_b - DO_b)/(Csat_t - DO_t)).min() < 0:
        DeltaC_lmd = 0.0001
    else:
        DeltaC_lmd = ((Csat_b - DO_b) - (Csat_t - DO_t))/np.log((Csat_b - DO_b)/(Csat_t - DO_t))
    
    #required kla to meet oxygen uptake rate
    kla_needed = O2_UR / DeltaC_lmd
    
    #Kolmogorov eddy disspitaion length scale from dissipated energy
    eps = b.power/(V_t/1000)/b.rho
    epsMax = 50*eps
    lambdaK = ((b.mu/b.rho)**3/epsMax)**0.25
    
    #find limits for each constraint
    indexAMM = min(AMM[AMM<c.limits[0]].shape[0],AMM.shape[-1]-1)
    indexLAC = min(LAC[LAC<c.limits[1]].shape[0],LAC.shape[-1]-1)    
    indexpCO2 = min(pCO2[pCO2<c.limits[2]].shape[0],pCO2.shape[-1]-1)
    
    klaRatio = kla_needed/kla_theory
    indexkla = min(klaRatio[klaRatio<1].shape[0],klaRatio.shape[-1]-1)
    
    mixing = tau_mix*kla_needed
    indexmixing = min(mixing[mixing<1].shape[0],mixing.shape[-1]-1)
    
    indexvolfrac = min(volFrac[volFrac<0.25].shape[0],volFrac.shape[-1]-1)
    indexlambdaK = min(lambdaK[lambdaK>=c.limits[3]].shape[0],lambdaK.shape[-1]-1)
    indexustop = min(u_s_top[u_s_top<=0.006].shape[0],u_s_top.shape[-1]-1)
    indexvolume = min(V_t[V_t<=0.8*b.wv].shape[0],V_t.shape[-1]-1)
    
    #report time and yield for each constraint limit
    limitsAMM = (t[indexAMM],gDW[indexAMM]/c.dmf/V_t[indexAMM])
    limitsLAC = (t[indexLAC],gDW[indexLAC]/c.dmf/V_t[indexLAC])
    limitspCO2 = (t[indexpCO2],gDW[indexpCO2]/c.dmf/V_t[indexpCO2])
    limitskla = (t[indexkla],gDW[indexkla]/c.dmf/V_t[indexkla])
    limitsmixing = (t[indexmixing],gDW[indexmixing]/c.dmf/V_t[indexmixing])
    limitsvolfrac = (t[indexvolfrac],gDW[indexvolfrac]/c.dmf/V_t[indexvolfrac])
    limitslambdaK = (t[indexlambdaK],gDW[indexlambdaK]/c.dmf/V_t[indexlambdaK])
    limitsustop = (t[indexustop],gDW[indexustop]/c.dmf/V_t[indexustop])
    limitsvolume = (t[indexvolume],gDW[indexvolume]/c.dmf/V_t[indexvolume])
    
    #constract dataframe with relevant constraints and parameters over time
    dataframe = pd.DataFrame({'Time [hr]': t, 'Volume [L]': V_t, 'Mixing Time [s]': tau_mix, 'Required kLa [1/s]': kla_needed \
                             , 'Theoretical kLa [1/s]': kla_theory, 'Required/Theoretical kLa [-]': kla_needed/kla_theory \
                             , 'Required kLa * Mixing Time [-]': kla_needed*tau_mix, 'pCO2 [mbar]': pCO2, 'Ammonia [mmol/L]': AMM \
                             , 'Lactate [mmol/L]': LAC, 'Volume Fraction [-]': volFrac, 'lambda_k [m]': lambdaK \
                              , 'Superficial Gas Top [m/s]': u_s_top, 'Cell Density [wet g/L]': gDW/c.dmf/V_t})
    
    #construct list of constraining parameters and list of time and yield limits of each 
    limitsDescriptList = ['ammonia', 'lactate', 'CO2', 'kla', 'mixing', 'volume fraction', 'hydrodynamic stress', 'superficial velocity', 'volume']
    limitsList = [limitsAMM, limitsLAC, limitspCO2, limitskla, limitsmixing, limitsvolfrac, limitslambdaK, limitsustop, limitsvolume]
    
    #construct a dictionary of parameters at the limited production time. Unused at present, left in for reference 
    tlimitindex=min(indexLAC,indexAMM,indexpCO2,indexkla,indexmixing,indexvolfrac,indexlambdaK,indexustop,indexvolume)
    paramsAtLimit={'Time':t[tlimitindex],'Cell Mass':gDW[tlimitindex]/c.dmf/V_t[tlimitindex],'Lactate':LAC[tlimitindex],'Ammonia':AMM[tlimitindex],'CO2':pCO2[tlimitindex],\
                        'kla ratio':klaRatio[tlimitindex],'mixing':mixing[tlimitindex],'volume fraction':volFrac[tlimitindex],'hydro stress':lambdaK[tlimitindex],\
                            'superficial velocity':u_s_top[tlimitindex],'volume':V_t[tlimitindex]}
    
    return (dataframe,limitsDescriptList,limitsList)



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
    dens = np.linspace(nslims[0],nslims[1],count)
    
    #make 3D and 2D mesh grids to store date
    rr, ss, dd = np.meshgrid(rpms, supers, dens, indexing='ij')
    rm, sm = np.meshgrid(rpms, supers, indexing='ij')
    
    #max yields over all n_s values - initialise to zero valued meshgrids
    overallYieldMAX = rm*0
    overallDurationMAX = rm*0
    nsMAX = rm*0
    lactateYieldMAX = rm*0
    ammoniaYieldMAX = rm*0
    pCO2YieldMAX = rm*0
    klaYieldMAX = rm*0
    mixingYieldMAX = rm*0
    volFracYieldMAX = rm*0
    lambdaKYieldMAX = rm*0
    ustopYieldMAX = rm*0
    volumeYieldMAX = rm*1e-6
    
    #yields over all rpm/u_s/n_s combinations - initialise to zero valued meshgrids
    ammoniaYield = ss*0
    lactateYield = ss*0
    pCO2Yield = ss*0
    klaYield = ss*0
    mixingYield = ss*0
    volFracYield = ss*0
    lambdaKYield = ss*0
    ustopYield = ss*0
    volumeYield = ss*0
    overallYield = ss*0
    overallDuration = ss*0

    #loop through each combination of rpm/u_s/n_s
    for i in range(count):
        for j in range(count):
            for k in range(count):
                # treat xv[i,j], yv[i,j]
                b.n = rr[i,j,k]
                b.u_s = ss[i,j,k]
                b.ns = dd[i,j,k]
                b.calctipSpeed()
                b.calcpower()
                b.calctau()
                (o,p,q) = prodPerf(b,c,time1)
            
                #determine overall yield and duration
                overallYield[i,j,k] = min(q[0][1],q[1][1],q[2][1],q[3][1],q[4][1],q[5][1],q[6][1],q[7][1],q[8][1])
                overallDuration[i,j,k] = min(q[0][0],q[1][0],q[2][0],q[3][0],q[4][0],q[5][0],q[6][0],q[7][0],q[8][0])

                #determine specific yields
                ammoniaYield[i,j,k] = q[0][1]
                lactateYield[i,j,k] = q[1][1]
                pCO2Yield[i,j,k] = q[2][1]
                klaYield[i,j,k] = q[3][1]
                mixingYield[i,j,k] = q[4][1]
                volFracYield[i,j,k] = q[5][1]
                lambdaKYield[i,j,k] = q[6][1]
                ustopYield[i,j,k] = q[7][1]
                volumeYield[i,j,k] = q[8][1]

                #determine max yields per constraint over all n_s values
                if (overallYield[i,j,k]>overallYieldMAX[i,j]):
                    overallYieldMAX[i,j]=overallYield[i,j,k]
                    overallDurationMAX[i,j]=overallDuration[i,j,k]
                    nsMAX[i,j] = dd[i,j,k]
                    
                if (lactateYield[i,j,k]>lactateYieldMAX[i,j]):
                    lactateYieldMAX[i,j]=lactateYield[i,j,k]
                    
                if (ammoniaYield[i,j,k]>ammoniaYieldMAX[i,j]):
                    ammoniaYieldMAX[i,j]=ammoniaYield[i,j,k]
                    
                if (pCO2Yield[i,j,k]>pCO2YieldMAX[i,j]):
                    pCO2YieldMAX[i,j]=pCO2Yield[i,j,k]
                    
                if (klaYield[i,j,k]>klaYieldMAX[i,j]):
                    klaYieldMAX[i,j]=klaYield[i,j,k]
                    
                if (mixingYield[i,j,k]>mixingYieldMAX[i,j]):
                    mixingYieldMAX[i,j]=mixingYield[i,j,k]
                
                if (volFracYield[i,j,k]>volFracYieldMAX[i,j]):
                    volFracYieldMAX[i,j]=volFracYield[i,j,k]
                
                if (lambdaKYield[i,j,k]>lambdaKYieldMAX[i,j]):
                    lambdaKYieldMAX[i,j]=lambdaKYield[i,j,k]
                
                if (ustopYield[i,j,k]>ustopYieldMAX[i,j]):
                    ustopYieldMAX[i,j]=ustopYield[i,j,k]
                
                if (volumeYield[i,j,k]>volumeYieldMAX[i,j]):
                    volumeYieldMAX[i,j]=volumeYield[i,j,k]
    
    if graphs:
        #remove nasty bit from duration
        blank = overallYieldMAX > 1


        plt.figure(figsize=(12,12))
        plt.subplot(4,3,1)
        plt.contourf(rm, sm, overallYieldMAX,levels=np.linspace(overallYieldMAX.min(),overallYieldMAX.max(),10))
        plt.colorbar()
        plt.title('Overall Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.subplot(4,3,2)
        plt.contourf(rm, sm, overallDurationMAX,levels=np.linspace(0.99*overallDurationMAX.min(),1.01*overallDurationMAX.max(),10))
        plt.colorbar()
        plt.title('Duration [hrs]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")


        plt.subplot(4,3,3)
        plt.contourf(rm, sm, nsMAX,levels=np.linspace(0.99*nsMAX.min(),1.01*nsMAX.max(),10))
        plt.colorbar()
        plt.title('Starting Cell Density [1/mL]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.subplot(4,3,4)
        plt.contourf(rm, sm, klaYieldMAX,levels=np.linspace(0.99*klaYieldMAX.min(),1.01*klaYieldMAX.max(),10))
        plt.colorbar()
        plt.title('Oxygen Mass Transfer\n Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.subplot(4,3,5)
        plt.contourf(rm, sm, lambdaKYieldMAX,levels=np.linspace(0.99*lambdaKYieldMAX.min(),1.01*lambdaKYieldMAX.max(),10))
        plt.colorbar()
        plt.title('Cell Stress\n Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")
        
        plt.subplot(4,3,6)
        plt.contourf(rm, sm, ustopYieldMAX,levels=np.linspace(0.99*ustopYieldMAX.min(),1.01*ustopYieldMAX.max(),10))
        plt.colorbar()
        plt.title('Superficial Velocity Top \nLimited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")
        
        plt.subplot(4,3,7)
        plt.contourf(rm, sm, pCO2YieldMAX,levels=np.linspace(0.99*pCO2YieldMAX.min(),1.01*pCO2YieldMAX.max(),10))
        plt.colorbar()
        plt.title('CO2 Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.subplot(4,3,8)
        plt.contourf(rm, sm, mixingYieldMAX,levels=np.linspace(0.99*mixingYieldMAX.min(),1.01*mixingYieldMAX.max(),10))
        plt.colorbar()
        plt.title('Mixing Limited Yield [g/L wet]')
        plt.xlabel("Agitation [RPM]")
        plt.ylabel("Aeration Superficial Velocity [m/s]")

        plt.tight_layout()
        plt.savefig('ConstraintYieldGraphs.png')
        plt.show()

    #construct dataframe to provide overall and per constraint maximum yields
    dataframe = pd.DataFrame({'Constraint': ['overall','ammonia','lactate','CO2','kla','mixing','volume fraction','hydrodynamic stress','superficial velocity','volume'],\
                              'Maximum Yield [g/L wet]':[overallYieldMAX.max(),ammoniaYieldMAX.max(),lactateYieldMAX.max(),pCO2YieldMAX.max(),klaYieldMAX.max(),\
                                mixingYieldMAX.max(),volFracYieldMAX.max(),lambdaKYieldMAX.max(),ustopYieldMAX.max(),volumeYieldMAX.max()]})
    
    #return ([overallYieldMAX.max(),lactateYieldMAX.max(),ammoniaYieldMAX.max(),pCO2YieldMAX.max(),\
    #        klaYieldMAX.max(),mixingYieldMAX.max(),volFracYieldMAX.max(),lambdaKYieldMAX.max(),\
    #        ustopYieldMAX.max(),volumeYieldMAX.max(),]) #make this a dataframe
    return (dataframe)    
###############