# Chrysopelea
![Logo](/flogo.png)\
A simple script for apply LSE to GPS data.
It was developed for the final project of the ICTP diploma/PhD Space Geodesy course.

The procedure implemented is a semplified version of the one described in\
Barzaghi, R., Borghi, A. Theory of second order stationary random processes applied to GPS coordinate time-series. *GPS Solut* **22**, 86 (2018). https://doi.org/10.1007/s10291-018-0748-4

## The implemented procedure 
 Using the 'color noise' stochastic model, each component of the data has been processed with the following scheme:

1.  In a pre-processing phase the data are interpolated to remove possible data gap;

2.  Least squares are applied using the a white noise covariance matrix and the following deterministic model:\
    x(t)=A0+Av*t+ΣA1(T)*sin(2πt/T)+ΣA2(T)*sin(2πt/T)+ΣAco(Tco)*H(t-Tco)\
    with *T* periodicity within the signal, *T*<sub>*co*</sub> time of known discontinuities in the signals (due to coseismic displacements, instrumentation failures and changes, etc.).

3. The periodicities of the signals are derived from a periodogram of the residuals: initially, the data are fitted with a simplified linear regression model and the periodogram of the residual of the least squares fitting is computed, to find out the most significant periodicities in the time series. These periodicity are introduced one at a time until the least squares fitting doesn't improve.\
    The data are initially assumed uncorrelated using a white noise stochastic model.

4.  The least squares residuals *ϵ*(*t*) are computed and checked for the stationary hypothesis using the KPSS test;

5.  The empirical Auto-Covariance Function *α*(*τ*) of the residuals is computed and it is used to define a "color noise" stochastic model: to avoid numerical problem, due to the inversion of the covariance matrix of the data, when estimating the parameters, the auto-covariance function *α*(*τ*) is fitted using a positive definite function.

5.  Least squares are applied using the updated stochastic model to evaluate the parameters **x̂**.

6.  The covariance matrix *C*<sub>**x̂**x̂</sub> of the parameters is computed.

The "white noise" stochastic model is computed in the same way but skipping steps 4-6.

## The data
The script has been developed to work with 24 hour final solutions (.tenv3) files from the Nevada Geodetic Laboratory.\
It also requires the NGL's Database of Potential Step Discontinuities (steps.txt).
As a result of a correct run, the script produces an output file with the results with the same name of the station and extension .res.\
To run correctly the script the name of the station has to be passed as an argument at the start, e.g.
```python SGprojBB.py AZ00```.
