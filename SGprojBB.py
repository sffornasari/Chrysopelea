"""
This script performs a simplified version of the procedures described in
Barzaghi, R., Borghi, A. Theory of second order stationary random processes applied to GPS coordinate time-series. GPS Solut 22, 86 (2018). https://doi.org/10.1007/s10291-018-0748-4
This script requires the name of the GPS station to be studied as a parameter when launched (e.g. >>>python SGprojBB.py AB07)
"""

import sys
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import linalg
from scipy.optimize import curve_fit, OptimizeWarning
from statsmodels.tsa.stattools import kpss, acf
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', InterpolationWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', OptimizeWarning)

#Data structure
class envdata:
    def __init__(self):
        self.site = None
        #self.YYMMMDD = None 
        self.yyyy_yyyy = None
        self.MJD = None
        self.week = None 
        self.d = None
        self.reflon = None
        self.e0 = None
        self.east = None
        self.n0 = None
        self.north = None
        self.u0 = None
        self.up = None
        self.ant = None
        self.sig_e = None
        self.sig_n = None
        self.sig_u = None
        self.corr_en = None
        self.corr_eu = None
        self.corr_nu = None
        self.disclist = []
        self.seisdisclist = []
        self.comp_e = self.Comp()
        self.comp_e.comp = 'E'
        self.comp_n = self.Comp()
        self.comp_n.comp = 'N'
        self.comp_u = self.Comp()
        self.comp_u.comp = 'U'   
    
    class Comp:
        def __init__(self):
            self.comp = None
            self.periodlist = []
            self.A = None
            self.C = None
            self.xh = None
            self.res = None
            self.acf = None
            self.Chh = None
    
    #Data "reading"
    def data(self, dataframe):
        self.site = dataframe.iloc[0]['site']
        #self.YYMMMDD = dataframe['YYMMMDD'].to_numpy() 
        self.yyyy_yyyy = dataframe['yyyy.yyyy'].to_numpy()
        self.MJD = dataframe['__MJD'].to_numpy()
        self.week = dataframe['week'].to_numpy() 
        self.d = dataframe['d'].to_numpy()
        self.reflon = dataframe['reflon'].to_numpy()
        self.e0 = dataframe['_e0(m)'].to_numpy()
        self.east = dataframe['__east(m)'].to_numpy()
        self.n0 = dataframe['____n0(m)'].to_numpy()
        self.north = dataframe['_north(m)'].to_numpy()
        self.u0 = dataframe['u0(m)'].to_numpy()
        self.up = dataframe['____up(m)'].to_numpy()
        self.ant = dataframe['_ant(m)'].to_numpy()
        self.sig_e = dataframe['sig_e(m)'].to_numpy()
        self.sig_n = dataframe['sig_n(m)'].to_numpy()
        self.sig_u = dataframe['sig_u(m)'].to_numpy()
        self.corr_en = dataframe['__corr_en'].to_numpy()
        self.corr_eu = dataframe['__corr_eu'].to_numpy()
        self.corr_nu = dataframe['__corr_nu'].to_numpy()
    
    #Perform basic pre-processing operations (e.g. remove data gap)
    def pre_proc(self):
        MJDrange = range(self.MJD[0], self.MJD[-1]+1)
        for attr, value in self.__dict__.items():
            if attr in ['yyyy_yyyy', 'week', 'd', 'reflon', 'e0', 'east', 'n0', 'north', 'u0', 'up', 'ant', 'sig_e', 'sig_n', 'sig_u', 'corr_en', 'corr_eu', 'corr_nu']:
                f = interp1d(self.MJD, value)
                setattr(self, attr, f(MJDrange))
        self.MJD = MJDrange
        disclist = []
        for tdisc in self.disclist:
            if tdisc in self.MJD:
                disclist.append(tdisc)
        self.disclist = disclist
        for y in [self.east, self.north, self.up]:
            sos = signal.butter(4, (1/len(y), 0.5), 'bandpass', output='sos')
            y = signal.sosfiltfilt(sos, y)

    #Define the design matrix
    def design_matrix(self, chan=None):
        if chan == None:
            chan=[self.comp_e, self.comp_n, self.comp_u]
        for cha in chan:
            A = np.zeros((len(self.MJD), 2+2*len(cha.periodlist)+len(self.disclist)))
            for i in range(len(self.MJD)):
                t = self.MJD[i]
                A[i][0:2]=(1, t)
                for j in range(len(cha.periodlist)):
                    omega = 2*np.pi/cha.periodlist[j]
                    A[i][2+2*j] = np.sin(omega*t)
                    A[i][2+2*j+1] = np.cos(omega*t)
                disc_thetas = [] 
                for tdisc in self.disclist:
                    disc_thetas.append(np.heaviside(t - tdisc, 1))
                    if t-tdisc < 0:
                        value = 0
                    else:
                        value=1
                A[i][2*len(cha.periodlist)+2:] = disc_thetas
            cha.A = A
    
    #Perform the LSE
    def LSE(self, cha_y=None):
        if cha_y == None:
            cha_y=[[self.comp_e, self.east], [self.comp_n, self.north], [self.comp_u, self.up]]
        for cha, y in cha_y:
            A = cha.A
            C = cha.C
            Ci = linalg.pinv(C)
            AtCi = A.T.dot(Ci)
            d = 0
            xh = linalg.pinv(AtCi.dot(A)).dot(AtCi).dot(y-d)
            cha.xh = xh
    
    #Evaluates the residuals
    def residuals(self, cha_y=None):
        if cha_y == None:
            cha_y=[[self.comp_e, self.east], [self.comp_n, self.north], [self.comp_u, self.up]]
        for cha, y in cha_y:
            cha.res = cha.A.dot(cha.xh) - y
    
    #Evaluates the periodicities in the signal
    def periodicities(self):
        for cha, y in [[self.comp_e, self.east], [self.comp_n, self.north], [self.comp_u, self.up]]:
            olderr = np.sum(cha.res**2)
            while True:
                f, P = signal.periodogram(cha.res, 1, window ='hamming', scaling='spectrum')
                cha.periodlist.append(1/f[np.argmax(P)])
                self.design_matrix([cha])
                self.LSE([[cha, y]])
                self.residuals([[cha, y]])
                if np.sum(cha.res**2)>olderr:
                    cha.periodlist = cha.periodlist[:-1]
                    self.design_matrix([cha])
                    self.LSE([[cha, y]])
                    self.residuals([[cha, y]])
                    break
                else:
                    olderr = np.sum(cha.res)
   
    #Perform a KPSS test on the residuals and, if failed, terminate the script
    def KPSS_test(self):
        for cha in [self.comp_e, self.comp_n, self.comp_u]:
            statistic, p_value, n_lags, critical_values = kpss(cha.res)
            if p_value < 0.05:
                print(f'The {cha.comp} series is not stationary')
                print(f'KPSS_c p-value:{p_value}')
                sys.exit(0)

    #Evaluate the ACF on the residuals and fitting with positive definite function
    def resACF(self):
        for cha in [self.comp_e, self.comp_n, self.comp_u]:
            eacf = acf(cha.res, nlags=len(cha.res)-1)
            N = 0
            c = 0
            for i in range(len(eacf)-1):
                if np.sign(eacf[i])!= np.sign(eacf[i+1]):
                    if c == 0:
                        c = 1
                    elif c==1:
                        N = i
                        break
            def ep(t, a, b, c):
                return a*np.exp(-b*t)*(1-c*t**2)
            def gp(t, a, b, c):
                return a*np.exp(-b*t**2)*(1-c*t**2)
            def ec(t, a, b, c):
                return a*np.exp(-b*t)*np.cos(1-c*t)
            def gc(t, a, b, c):
                return a*np.exp(-b*t**2)*np.cos(c*t)
            def e(t, a, b):
                return a*np.exp(-b*t)
            def g(t, a, b):
                return a*np.exp(-b*t**2)
            def err(x, y):
                return np.sum((x-y)**2)
            old_err = np.inf
            t = np.array(self.MJD) - self.MJD[0]
            for f in [ep, gp, ec, gc, e, g]:
                popt, pcov = curve_fit(f, t[1:N], eacf[1:N])
                sacf = f(t, *popt)
                sacf[0] = 1
                if err(sacf[1:N], eacf[1:N])<old_err:
                    cha.acf = sacf
                    old_err = err(sacf[1:N], eacf[1:N])   
    
    #Define or update the covariance matrix
    def C_update(self):
        for cha, sig in [[self.comp_e, self.sig_e], [self.comp_n, self.sig_n], [self.comp_u, self.sig_u]]:
            if cha.acf is None:
                cha.C = np.diag(sig**2)
            else:
                N = len(cha.acf)
                C = np.zeros((N,N))
                for i in range(N):
                    for j in range(N):
                        tau = np.abs(i-j)
                        #if tau != 0:
                        C[i,j] = cha.acf[tau]
                cha.C = C + np.diag(sig**2)
    
    #Evaluate the covariance matrix for the estimated parameters
    def C_est(self):
        for cha in [self.comp_e, self.comp_n, self.comp_u]:
            varh = cha.res.T.dot(np.linalg.pinv(cha.C)).dot(cha.res)/(len(cha.res)-len(cha.xh))
            Chh = varh*np.linalg.pinv(cha.A.T.dot(np.linalg.pinv(cha.C)).dot(cha.A))
            cha.Cest = Chh
            array_sum = np.sum(Chh)
            array_has_nan = np.isnan(array_sum)
    
    #Show on screen and save on file the results
    def results(self):
        with open(site+'.res', 'w', encoding="utf-8") as out:
            out.write(f'Site: {env.site}\n')
            out.write(f'Start: {env.MJD[0]} MJD\n')
            out.write(f'End: {env.MJD[-1]} MJD\n')
            out.write('Stochastic model: x(t)=A0+Av*t+\u03A3A1(T)*sin(2\u03C0t/T)+\u03A3A2(T)*sin(2\u03C0t/T)+\u03A3Aco(Tco)*H(t-Tco)\n')
            for cha, chaname in [[self.comp_e, 'EAST'], [self.comp_n, 'NORTH'], [self.comp_u, 'UP']]:
                sd = np.sqrt(np.diag(cha.Cest))
                varlist = ['A0', 'Av']
                for P in cha.periodlist:
                    varlist.append(f'A1(T={np.round(P, 1)} d)')
                    varlist.append(f'A2(T={np.round(P, 1)} d)')
                for T in self.disclist:
                    varlist.append(f'ACO(MJD={T})')
                print(f'{chaname} COMPONENT RESULTS:')
                out.write(f'{chaname} COMPONENT RESULTS:\n')
                for res in enumerate(cha.xh):
                    print(f'{varlist[res[0]]} : {np.round(res[1], 5)} \u00B1 {np.round(sd[res[0]], 5)} m')
                    out.write(f'{varlist[res[0]]} : {np.round(res[1], 5)} \u00B1 {np.round(sd[res[0]], 5)} m\n')
                print('='*50)
                out.write('='*50+'\n')

#Main program
if __name__ == '__main__':

    #Proper call of the script
    if len(sys.argv) == 1:
    print("You must specify the station name!")
    sys.exit(0)
    
    #Specify the number of observation to use (from the newest)
    lastdata = 1817
    
    #Color-noise stochastic model?
    cncheck = True

    #Reading data from the file
    site = sys.argv[1]
    datafile = site+'.tenv3'
    df = pd.read_csv(datafile, header=0, skipinitialspace=True, delim_whitespace=True)
    env = envdata()
    env.data(df.tail(lastdata))

    #Defining known discontinuities (from Database of Potential Step Discontinuities)
    columnlist = ['site', 'date']
    try:
        steps = pd.read_csv('steps.txt', header=None, skipinitialspace=True, delim_whitespace=True, usecols=[0,1,2])
    except:
        print("You must have the Database of Potential Step Discontinuities file in the current folder!")
    monthlist = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for index, row in steps.iterrows():       
        if row[0] == site:
            day = int(row[1][5:7])
            month = monthlist.index(row[1][2:5])+1
            year = int(row[1][0:2])
            year = year + 2000 if year<50 else year+1900
            disctime = int(np.round(367 * year - 7 * (year + (month + 9)/12)/4 + 275 * month/9 + day - 678986.5))
            if disctime not in env.disclist:
                env.disclist.append(disctime)
                if  ascheck == True and row[2] == 2:
                    env.seisdisclist.append(disctime)
    
    #Correcting the data for gaps (interpolated) & other preprocessing steps
    env.pre_proc()
    
    #White noise model LSE
    env.design_matrix()
    env.C_update()
    env.LSE()
    env.residuals()
    env.periodicities()
    
    #Color noise model LSE
    if cncheck == True:
        env.KPSS_test()
        
        env.resACF()
        env.C_update()
        env.LSE()
        env.residuals()
    
    #Covariance matrix of estimated parameters
    env.C_est()
    
    #Showing/Saving the results
    env.results()
    
    #Plotting the data and LSE fitted model
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
    
    ax1.plot(env.MJD, env.east)
    ax1.plot(env.MJD, env.comp_e.A.dot(env.comp_e.xh))
    ax1.set_ylabel('East [m]')
    ax1.grid()
    
    ax2.plot(env.MJD, env.north)
    ax2.plot(env.MJD, env.comp_n.A.dot(env.comp_n.xh))
    ax2.set_ylabel('North [m]')
    ax2.grid()
    
    ax3.plot(env.MJD, env.up)
    ax3.plot(env.MJD, env.comp_u.A.dot(env.comp_u.xh))
    ax3.set_ylabel('Up [m]')
    ax3.set_xlabel('MJD [d]')
    ax3.grid()
    
    plt.show()
