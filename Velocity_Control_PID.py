
''' LIBRARIES '''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' DEFINE MODEL '''

def vehicle(v,t,u):
    dv_dt = a       # v = u + at               
    return dv_dt    # Derivative

''' FOR ANIMATION'''

tf = 300.0                    # final time for simulation
nsteps = 301                  # number of time steps
delta_t = tf/(nsteps-1)       # how long is each time step?
ts = np.linspace(0,tf,nsteps) # linearly spaced time vector
step = np.zeros(nsteps)       # simulate step test operation
vs = np.zeros(nsteps)         # for storing the results 

v0 = 0.0                      # velocity initial condition 
sp = 2.0                      # set point 
Error = 0

''' PID = Kc*error + (Kc/tauI)*(integral of error) + (Kc*td)*(derivative of error)  '''

Kc = 0.1
tauI = 10.0
td = 0.1

sum_int = 0.0                  # integral of error 
sum_der = 0.0                  # derivative of error

es = np.zeros(nsteps)          # error
ies = np.zeros(nsteps)         # integral of error
sps = np.zeros(nsteps)         # set point store 

''' PLOT '''
plt.figure(1,figsize=(5,4))
plt.ion()
plt.show()

''' SIMULATE WITH 'ODEINT' '''

for i in range(nsteps-1):
	
    sps[i+1] = sp
    error = sp - v0
    es[i+1] = error

    sum_int = sum_int + error + delta_t
    sum_der = sum_der - error - delta_t

    a =  (Kc*error + Kc/tauI * sum_int + Kc*td*sum_der)
    
    # Limits of Acceleration    
    if a >= 3.0:
        a = 3.0
        sum_int = sum_int - error * delta_t
   
    ies[i+1] = sum_int
    step[i+1] = a 

    v = odeint(vehicle,v0,[0,delta_t],args=(a,))                  # calculates v(t)
    v0 = v[-1]                                                    # take the last value
    vs[i+1] = v0                                                  # store the velocity for plotting

    
    ''' ANIMATION '''

    plt.clf()
    plt.subplot(2,2,1)
    plt.plot(ts[0:i+1],vs[0:i+1],'b-',linewidth=3)
    plt.plot(ts[0:i+1],sps[0:i+1],'k--',linewidth=2)
    plt.ylabel('Velocity (m/s)')
    plt.legend(['Velocity','Set Point'],loc=2)
    plt.subplot(2,2,2)
    plt.plot(ts[0:i+1],step[0:i+1],'r--',linewidth=3)
    plt.ylabel('Acceleration (m/s^2)')    
    plt.legend(['Acceleration'])
    plt.subplot(2,2,3)
    plt.plot(ts[0:i+1],es[0:i+1],'b-', linewidth=3)
    plt.legend(['Error(SP-SV)'])
    plt.xlabel('Time (sec)')
    plt.subplot(2,2,4)
    plt.plot(ts[0:i+1],ies[0:i+1],'k--',linewidth=3)
    plt.legend(['Integral of Error'])
    plt.xlabel('Time (sec)')
    plt.pause(0.1)   
  
