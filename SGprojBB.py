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
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import kpss, acf
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', InterpolationWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)

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
        self.commonperiodlist = []
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
    
    def data(self, dataframe):
        self.site = dataframe['site'].head(0)
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
        discind = []
        for tdisc in self.disclist:
            if tdisc in self.MJD:
                disclist.append(tdisc)
                discind.append(self.MJD.index(tdisc))
        self.disclist = disclist
        #for cha in [self.comp_e, self.comp_n, self.comp_u]:
        for cha, y in [[self.comp_e, self.east], [self.comp_n, self.north], [self.comp_u, self.up]]:        
            cha.periodlist += self.commonperiodlist
            #y = signal.detrend(y, axis=- 1, bp=discind)

#Define the design matrix
    def design_matrix(self):
        for cha in [self.comp_e, self.comp_n, self.comp_u]:
            A = np.zeros((len(self.MJD), 2+2*len(cha.periodlist)+len(self.disclist)))
            for i in range(len(self.MJD)):
                t = self.MJD[i]
                A[i][0:2]=(1, t)
                for j in range(len(cha.periodlist)):
                    omega = 2*np.pi/cha.periodlist[j]
                    A[i][2+2*j] = np.sin(omega*t)
                    A[i][2+2*j+1] = np.cos(omega*t)
                A[i][2*len(cha.periodlist)+2:] = [np.heaviside(t - tdisc, 1) for tdisc in self.disclist]
            cha.A = A

#Perform the LSE
    def LSE(self):
        for cha, y in [[self.comp_e, self.east], [self.comp_n, self.north], [self.comp_u, self.up]]:
            A = cha.A
            C = cha.C
            Ci = linalg.pinv(C)
            AtCi = A.T.dot(Ci)
            d = 0
            xh = linalg.pinv(AtCi.dot(A), cond=1e-20).dot(AtCi).dot(y-d)
            cha.xh = xh

#Evaluates the residuals     
    def residuals(self):
        for cha, y in [[self.comp_e, self.east], [self.comp_n, self.north], [self.comp_u, self.up]]:
            cha.res = cha.A.dot(cha.xh) - y
          
#Perform a KPSS test on the residuals and, if failed, plot the periodogram of the residuals and exit the current run
    def KPSS_test(self):
        for cha in [self.comp_e, self.comp_n, self.comp_u]:
            statistic, p_value, n_lags, critical_values = kpss(cha.res)
            statistic_ct, p_value_ct, n_lags_ct, critical_values_ct = kpss(cha.res, regression='ct')
            if p_value < 0.05 or  p_value_ct < 0.05:
                if p_value < 0.05:
                    print(f'The {cha.comp} series is not stationary')
                    print(f'KPSS_c p-value:{p_value}')
                if p_value_ct < 0.05:
                    print(f'The {cha.comp} series is not trend-stationary')
                    print(f'KPSS_ct p-value:{p_value_ct}')
                    f, P = signal.periodogram(cha.res, 1, window ='flattop', scaling='density')
                    plt.semilogy(1/f, P)
                    plt.grid()
                    plt.show()
                    #cha.periodlist.append(1/f[np.argmax(P)])
                    #self.design_matrix()
                    #self.LSE()
                    #self.residuals()
                    #self.KPSS_test()
                print(f"The KPSS test failed: the {cha.comp} component residuals aren't stationary.")
                sys.exit(0)

#Evaluate the ACF on the residuals
    def resACF(self):
        for cha in [self.comp_e, self.comp_n, self.comp_u]:
            eacf = acf(cha.res, nlags=len(cha.res)-1)
            N = 15
            div = np.array([N]*len(eacf))
            div[:N]= [i+1 for i in range(N)]
            div[-N:]= [N-i for i in range(N)]
            sacf = np.convolve(eacf, np.ones(N), 'same')
            sacf = np.divide(sacf, div)
            cha.acf = sacf
            plt.plot(eacf)
            plt.plot(sacf)
            plt.show()

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
                        C[i,j] = cha.acf[tau]
                cha.C = C + np.diag(sig**2)
                if not np.all(np.linalg.eigvals(cha.C) >= 0):
                    print(f'The covariance matrix for the {cha.comp} component is not positive definite.')
                    sys.exit(0)

#Evaluate the covariance matrix for the estimated parameters
    def C_est(self):
        for cha in [self.comp_e, self.comp_n, self.comp_u]:
            varh = cha.res.T.dot(np.linalg.pinv(cha.C)).dot(cha.res)/(len(cha.res)-len(cha.xh))
            Chh = varh*np.linalg.pinv(cha.A.T.dot(np.linalg.pinv(cha.C)).dot(cha.A))
            cha.Cest = Chh

#Show on screen and save on file the results
    def results(self):
        with open(site+'.res', 'w', encoding="utf-8") as out:
            out.write(f'# site: {env.site}\n')
            out.write(f'# start: {env.MJD[0]}\n')
            out.write(f'# end: {env.MJD[-1]}\n')
            out.write('# stochastic model: x(t)=A0+Av*t+\u03A3A1(T)*sin(2\u03C0t/T)+\u03A3A2(T)*sin(2\u03C0t/T)+\u03A3Aco(Tco)*H(t-Tco)\n')
            for cha, chaname in [[self.comp_e, 'EAST'], [self.comp_n, 'NORTH'], [self.comp_u, 'UP']]:
                sd = np.sqrt(np.diag(cha.Cest))
                varlist = ['A0', 'Av']
                for P in cha.periodlist:
                    varlist.append(f'A1(T={P} d)')
                    varlist.append(f'A2(T={P} d)')
                for T in self.disclist:
                    varlist.append(f'ACO(MJD={T})')
                print(f'{chaname} COMPONENT RESULTS:')
                out.write(f'{chaname} COMPONENT RESULTS:\n')
                for res in enumerate(cha.xh):
                    print(f'{varlist[res[0]]} : {np.round(res[1], 5)} \u00B1 {np.round(sd[res[0]], 5)}')
                    out.write(f'{varlist[res[0]]} : {np.round(res[1], 5)} \u00B1 {np.round(sd[res[0]], 5)}\n')
                print('='*50)
                out.write('='*50+'\n')

#Main program
if __name__ == '__main__':
    #Argument check
    if len(sys.argv) == 1:
        print("You must specify the station name!")
        sys.exit(0)

    #Specify the number of observation to use (from the newest)
    lastdata = 1817
    #Specify periods of known periodicities common to all components
    commonperiods = [351.6/2, 351.6]

    #Reading data from the file
    site = sys.argv[1]
    datafile = site+'.tenv3'
    df = pd.read_csv(datafile, header=0, skipinitialspace=True, delim_whitespace=True)
    env = envdata()
    env.data(df.tail(lastdata))

    #Defining known discontinuities
    columnlist = ['site', 'date']
    try:
        steps = pd.read_csv('steps.txt', header=None, skipinitialspace=True, delim_whitespace=True, usecols=[0,1])
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
    
    #Defining known periodicity
    env.commonperiodlist = commonperiods
    
    #Correcting the data for gaps (interpolated) & other preprocessing steps
    env.pre_proc()
    
    #1st LSE run with "white noise"
    env.design_matrix()
    env.C_update()
    env.LSE()
    env.residuals()
    
    #Performing KPSS test
    env.KPSS_test()
    
    #Evaluation of ACF on the residuals and update of the covariance matrix
    env.resACF()
    env.C_update()
    
#2nd LSE run with "colored noise"
    env.LSE()

#Evaluation of the covariance matrix for the estimated parameters
    env.C_est()

#Printing and saving of the results  
    env.results()
