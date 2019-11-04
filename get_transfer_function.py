import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def get_MODEL_params(Km_init,taum_init,thetam_init):

    # Import CSV data file
    # Column 1 = time (t)
    # Column 2 = input (u)
    # Column 3 = output (yp)
    data = np.loadtxt('data.txt', delimiter=',')

    t = data[:, 0].T
    u = data[:, 1].T
    yp = data[:, 2].T

    # number of steps
    nr_steps = len(t)

    # linear interpolation between every datapoint
    uf = interp1d(t,u)

    # define first-order plus dead-time approximation
    def fopdt(y,t,uf,Km,taum,thetam):
        # arguments
        #  y      = output
        #  t      = time
        #  uf     = interpolated inputdata
        #  Km     = gain
        #  taum   = model time constant
        #  thetam = model dead time
        # time-shift u
        try:
            if (t-thetam) <= 0:
                um = uf(0.0)
            else:
                um = uf(t-thetam)
        except:
            um = u[0]

        # calculate derivative
        dydt = (-(y-yp[0]) + Km * (um-u[0]))/taum
        return dydt

    # simulate FOPDT model with x=[Km,taum,thetam]

    def sim_model(x):
        # input arguments
        Km = x[0]
        taum = x[1]
        thetam = x[2]

        # storage for model values
        ym = np.zeros(nr_steps)  # model

        ym[0] = yp[0] # initial condition
        # loop through time steps
        for i in range(0,nr_steps-1):
            ts = [t[i],t[i+1]]
            y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
            ym[i+1] = y1[-1]

        return ym

    # define objective
    def objective(x):

        # simulate model
        ym = sim_model(x)

        # calculate objective
        obj = 0.0
        for i in range(len(ym)):
            obj = obj + (ym[i]-yp[i])**2

        return obj

    # initial guesses
    x0 = np.zeros(3)
    x0[0] = Km_init # Km
    x0[1] = taum_init # taum
    x0[2] = thetam_init # thetam

    # optimize Km, taum, thetam
    bnds = ((-1.0e10, 1.0e10), (0.0001, 1.0e10), (0.0, 5.0))
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
    x = solution.x

    Km=x[0]
    taum=x[1]
    thetam=x[2]

    # calculate model with updated parameters
    ym1 = sim_model(x0)
    ym2 = sim_model(x)
    # plot results
    plt.figure()

    plt.plot(t, ym1, 'b--', linewidth=2, label='Initial Guess')
    plt.plot(t, ym2, 'r-', linewidth=3, label='Optimized FOPDT')
    plt.plot(t, yp, 'k', linewidth=2, label='Process Data')
    plt.ylabel('Output')
    plt.legend(loc='best')
    print(f"MODEL: {Km}, {taum}, {thetam} ")
    plt.show()
    return Km, taum, thetam

get_MODEL_params(0.00000001,0.00000001,0)