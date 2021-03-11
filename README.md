# Chrysopelea
![GitHub Logo](/logo.png)\
A simple script for apply LSE to GPS data.
It was developed for the final project of the ICTP diploma/PhD Space Geodesy course.

The procedure implemented is a semplified version of the one described in\
Barzaghi, R., Borghi, A. Theory of second order stationary random processes applied to GPS coordinate time-series. *GPS Solut* **22**, 86 (2018). https://doi.org/10.1007/s10291-018-0748-4

## The implemented procedure 
Each component of the data has been processed with the following scheme

1.  In a pre-processing phase the data are interpolated to remove
    possible data gap;

2.  Least squares are applied using the a white noise covariance matrix and the following deterministic model:\
    x(t)=A0+Av*t+ΣA1(T)*sin(2πt/T)+ΣA2(T)*sin(2πt/T)+ΣAco(Tco)*H(t-Tco)\
    with *T* periodicity within the signal, *T*<sub>*co*</sub> time of known discontinuities in the signals (due to coseismic displacements, instrumentation failures and changes, etc.).\
    As it is obvious from the stochastic model, the data are initially
    assumed uncorrelated.

3.  The least squares residuals *ϵ*(*t*) are computed and checked for
    the stationary hypothesis using the KPSS test: although the
    automatic procedure to check the periodogram of the residual to take
    into consideration the most significant periodicities in the time
    series has been implemented it was not used due to the poor results
    obtained, probably due to an incomplete pre-processing phase;

4.  The empirical Auto-Covariance Function *α*(*τ*) of the residuals is computed and the covariance matrix of the data is re-computed using the ACF to define the out-of-diagonal elements.
    To avoid numerical problem, due to the inversion of the covariance matrix of the data, when estimating the parameters, the auto-covariance function *α*(*τ*) is smoothed with a running average and the covariance matrix is checked to be positive semi-definite.

5.  Least squares are applied using the updated stochastic model to evaluate the parameters **x̂**;

6.  The covariance matrix *C*<sub>**x̂**x̂</sub> of the parameters is computed.

## The data
The script has been developed to work with 24 hour final solutions (.tenv3) files from the Nevada Geodetic Laboratory.\
It also requires the NGL's Database of Potential Step Discontinuities (steps.txt).
As a result of a correct run, the script produces an output file with the results with the same name of the station and extension .res.\
To run correctly the script the name of the station has to be passed as an argument at the start, e.g.
```python SGprojBB.py AZ00```.
