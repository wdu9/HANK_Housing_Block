



# By William Du and Jamie Lenney
# comments next to imports are package versions
import numpy as np # 1.23.5
from numba import njit #  0.56.4
import matplotlib.pyplot as plt # '3.6.2'
import copy
# import sequence_jacobian as sj
#sequence-jacobian==1.0.0
from sequence_jacobian.utilities.discretize import nonlinspace
from sequence_jacobian import grids, interpolate
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import Continuous1D, ExogenousMaker, LogitChoice

#%% Dictionary of parameters


Ph = 16.0 # house price
cali = dict()
r_ss = 0.05/4 # interest rate
relHRss = .64 # share of households who are owners
cali = {'taste_shock': 1E-1, # taste shock standard deviation
        'vphi': 1.0, 
        'rstar': r_ss, # interest rate
        'eis': 0.5, # CRRA = 1/eis
        'vh1':0,
        'rho_z': 0.95,  # persistence of AR process for idiosyncratic income
        'sd_z': 0.5,  # standard deviation of AR process
        'n_z': 5, # number of labor income prod gridpoints
        
        
        
        'min_a': -Ph*.9, # minimum grid point.
        'max_a': 250,  # maximum grid point
        'n_a': 100, # number of gridpoints
        
        
        'frisch':0.5, # frisch elasticity
        'piw':0.0, # wage inflation
        'kappaw':0.10, # wage stickiness
        'Y': 1.0, # output
        'Z': .6,  # TFP
        'pi': 0.0,
        'mu': 1.2, # markup
        'kappa': 0.1,  # price stickiiness
        'kappaw': 0.1,
        'phi': 1.5, # taylor rule coefficient on inflation
        'rhotax':0.8,
        'gammatax':6,
        'epsilon_pr': 1.0,

        
        'B':1, # government bonds
        'Bss': 1.0, # steady state government bonds
        'Taxss': .015, # steady state taxes
        'Yss': 1.0, # steady state ouput
        'wss': .5, # steady state wage
        
        'rbss': r_ss, 
        'relHRss': relHRss,
        
        'gammay':1.5,
        
        
        'rhoph': 0.9,'epsph':0.0,'rhor':0.6,'epsr':0,
                'alphah':1.0,
                
                
                
        'hbar': 0.8704, # this hbar is needed to target share of owners to be 64%
        'deltah': 0.05,
        'ph': Ph, # house price
        'phss':Ph, # steady state house price
        'ph_wealth': Ph,  # this is to decompose wealth effects
        'transac':0.01*Ph, # transaction cost
        'kappah':0.75, # loan to value
        'relHR':.64,
        'MPC':0.0,    # Don't touch this. Need this to be zero to compute IMPCs   

        
        # Below are must be zero in eq. These are explicitly specified because solve_impulse_nonlinear requires it to be in the dict
        'i_res': 0.0,
        'nkpc_res': 0.0,
        'B_res': 0.0,
        'Tax_res': 0.0,
        'eq_arb':0.0,
        'asset_mkt': 0.0,
        'goods_mkt':0.0,
        'labor_mkt':0.0,
        'housing_mkt':0.0,}






#%%

z_grid, z_dist, z_markov = grids.markov_rouwenhorst(cali['rho_z'],cali['sd_z'],cali['n_z']) # generate labor productivity grid, distribution, and transition matrix

cali['Ladj']=np.sum(z_dist*z_grid)


def make_grids(rho_z, sd_z, n_z, min_a, max_a, n_a):
    
    z_grid, z_dist, z_markov = grids.markov_rouwenhorst(rho_z, sd_z, n_z)
    

    #def nonlinspace(amax, n, phi, amin=0):
        # If phi =1 then points are equidistant. If phi > 1 then points are bunched near min point. Large phi, dense near the min.
    a_grid = nonlinspace(1.0,n_a ,1.9, amin = 0.0) # normalized grid between zero and one with n_a number of points
        

    a_grid_r = expandx(a_grid,0.0,max_a) # renters grid
    a_grid_o = expandx(a_grid,min_a,max_a) # owners grid
    
    # this is end of period assets
    a_grid_exp_t1 = np.zeros((4,n_a))
    a_grid_exp_t1[0] = a_grid_o # own to own
    a_grid_exp_t1[1] = a_grid_r # own to rent
    a_grid_exp_t1[2] = a_grid_o # rent to own
    a_grid_exp_t1[3] = a_grid_r # rent to rent
    
    # this is beginning of period assets
    a_grid_exp_t0 = np.zeros((4,n_a))
    a_grid_exp_t0[0] = a_grid_o # own to own
    a_grid_exp_t0[1] = a_grid_o # own to rent
    a_grid_exp_t0[2] = a_grid_r # rent to own 
    a_grid_exp_t0[3] = a_grid_r # rent to rent

    return z_grid, z_dist, z_markov, a_grid , a_grid_exp_t0, a_grid_exp_t1,a_grid_r,a_grid_o



def labor_income(a_grid, z_grid, r, w, phss,ph, transac, pr,kappah,hours,Tax,Div,MPC,a_grid_exp_t0):

    divi=(Div)*z_grid/(np.sum(z_grid*z_dist)) # incidence rule for dividend
        
    y = z_grid[np.newaxis, :] * np.array([w, w, w, w ])[:, np.newaxis]*hours*(1-Tax)  +divi[np.newaxis,:]  + np.array([0,  ph - transac - pr,  -  ph - transac, - pr])[:, np.newaxis]  + np.array([MPC, MPC,  MPC, MPC])[:, np.newaxis] # on (n, z)
    # cash on hand today
    coh = (1 + r) * a_grid_exp_t0[:, np.newaxis, :] + y[..., np.newaxis]  # on (n, z, a)

    return y, coh


# Below is needed to set up a separate household problem in order to compute wealth effects
def labor_income_wealth_effect(a_grid, z_grid, r, w, phss, ph_wealth, transac, pr,kappah,hours,Tax,Div,MPC,a_grid_exp_t0):

    divi=(Div)*z_grid/(np.sum(z_grid*z_dist)) # incidence rule for dividend

    y = z_grid[np.newaxis, :] * np.array([w, w, w, w ])[:, np.newaxis]*hours*(1-Tax)  +divi[np.newaxis,:]  + np.array([0,  ph_wealth - transac - pr,  -  ph_wealth - transac, - pr])[:, np.newaxis]  + np.array([MPC, MPC,  MPC, MPC])[:, np.newaxis] # on (n, z)
    # cash on hand today
    coh = (1 + r) * a_grid_exp_t0[:, np.newaxis, :] + y[..., np.newaxis]  # on (n, z, a)

    return y, coh



## households states (n-labour supply,z-productivyt,a-liquid wealth)
def hh_init(coh, eis,a_grid_exp_t0,r,w,Tax,hours,z_grid,ph, vh1, vh2,alphah,beta):
        
    # this is to indicate owners in utility function
    H_vals = np.zeros_like(coh)
    H_vals[0] = H_vals[0]  + 1
    H_vals[0] = H_vals[2]  + 1
    
    y = z_grid[np.newaxis, :] * np.array([w, w, w, w ])[:, np.newaxis]*hours*(1-Tax)

    c_temp = r * a_grid_exp_t0[:, np.newaxis, :] +y[...,np.newaxis]
    

    V = util_c_h(c_temp, H_vals ,eis,vh1, vh2,alphah) / (1-beta)

    Va = np.empty_like(V)
    
    Va=marg_util_c_h(c_temp, H_vals ,eis,vh1, vh2,alphah)  
    
    return V, Va 


# tenure state is d= {own/own, own/rent, rent/own, rent/rent}


'''
states:
own  to own
own  to rent
rent to own
rent to rent

'''

def util_l(V,max_a,n_a,ph, kappah,coh, pr,transac):
# on (n| n_)

    flow_u = np.array([[0.0,        -np.inf,     0  ,   -np.inf  ],    # (O/O |  O/O) , (O/O | O/R) , (O/O | R/O) , (O/O | R/R)
                       [0.0,       -np.inf,     0 ,    -np.inf  ],     # (O/R |  O/O) , (O/R | O/R) , (O/R | R/O) , (O/R | R/R)
                       [-np.inf ,   0,    - np.inf  ,       0    ],    # (R/O |  O/O) , (R/O | O/R) , (R/O | R/O) , (R/O | R/R)
                       [-np.inf ,   0,     -np.inf   ,      0     ],]) # (R/R |  O/O) , (R/R | O/R) , (R/R | R/O) , (R/R | R/R)

   #(R/O | O/R) being nonzero indicates that the state where you solve Own to Rent, you will consider tomorrows possible value of solving a Rent to Own problem,

    num_states = 4
    # on (n| n_, z, a)
    shape = np.zeros((num_states, num_states,) + V.shape[1:])
    flow_u = flow_u[..., np.newaxis, np.newaxis] + shape
    
    # Below is to ensure that households who enter the rent/own state have enough assets such that when they purchase a house they dont have negative consumption
    flow_u[2,3,( coh[2]   < -ph*kappah  )] = -np.inf # rent/rent to rent/own
    flow_u[2,1,( coh[2]   < -ph*kappah  )] = -np.inf # own/rent to rent/own
  
    return flow_u



# consumpton savings stage conditional on housing decision, solved using EGM with upper envelope
def dcegm(V, Va, a_grid, coh, y, r, beta, eis,kappah , vh1, vh2,alphah,max_a,n_a,phss,a_grid_exp_t0,a_grid_exp_t1,ph): # ,a_grid_exp_t0 ,a_grid_exp_t1):
    """DC-EGM algorithm"""
    # use all FOCs on endogenous grid
    
    # Below is used for utility function to have owners receive utility for owning a home
    H_vals = np.zeros_like(coh) # n x z x a
    H_vals[0] = H_vals[0]  + 1

    W = beta * V                                                  # end-of-stage vfun
    uc_endo = beta * Va                                           # envelope condition
    
    # THE EULER BELOW WILL NEED TO BE CHANGED SLIGHTLY TO ACCOMODATE VALUES OF ALPHAH<1
    c_endo = uc_endo** (-eis)                                     # Euler equation
    a_endo = (c_endo + a_grid_exp_t1[:,np.newaxis,:] - y[:,:,np.newaxis] ) / (1 + r) 

    # interpolate with upper envelope, compute unconstrained values and policies
    V_uc, c_uc, a_uc = upperenv_vec(W, a_endo, coh, a_grid_exp_t0,a_grid_exp_t1,  eis ,vh1, vh2,alphah,ph,kappah)

    constraint_renters = np.min(a_grid_exp_t1[3])
    #enforce borrowing constraint
    V, c, a_x  = constrain_policies(constraint_renters, V_uc, c_uc, a_uc , a_grid_exp_t0,coh,eis, vh1, vh2, alphah,W,ph,kappah,a_grid_exp_t1)

    # normalize asset policy
    a = np.zeros_like(a_x)
    for i in range(len(a_grid_exp_t1)):
        min_a_i=np.min(a_grid_exp_t1[i,:])
        max_a_i=np.max(a_grid_exp_t1[i,:])
        a[i,:,:] = normx(a_x[i,:,:],min_a_i,max_a_i)  
    
    # update Va on exogenous grid
    
    uc = marg_util_c_h(c, H_vals, eis, vh1, vh2,alphah)           # Computer marginal utility
    Va = (1 + r) * uc                                             # envelope condition

    return V, Va, a, c






@njit
def upperenv_vec(W, a_endo, coh, a_grid_exp_t0, a_grid_exp_t1, eis ,vh1, vh2,alphah, P, kappah): # a_grid here is taking in the normalized grid, even though when called I directly specify the tiled grid
    """Interpolate value function and consumption to exogenous grid."""
    n_n, n_z, n_a = W.shape 
    a = np.zeros((n_n,n_z,n_a)) # asset policy
    c = np.zeros((n_n,n_z,n_a)) # consumption policy
    V = -np.inf * np.ones((n_n,n_z,n_a))# Value functions

 
    for i_n in range(n_n): # ownership state
        
        for i_z in range(n_z): # productivity state
            
            for ja in range(n_a - 1): # loop over a_endo, generating segments of the a_endo grid
                
                
                a_low, a_high = a_endo[i_n,i_z, ja], a_endo[i_n,i_z, ja + 1] 
                W_low, W_high = W[i_n,i_z, ja], W[i_n,i_z, ja + 1]
                ap_low, ap_high = a_grid_exp_t1[i_n,ja], a_grid_exp_t1[i_n, ja + 1]
                    
               # loop over exogenous asset grid (increasing) 
                for ia in range(n_a):  
            
                    acur = a_grid_exp_t0[i_n,ia]  # this point is for looping over the segment for this iteration [a_grid is the grid you transition from ]
                                                  # The bounds of the segments for this iteration are a_low,a_high
                                                  # we use this agrid because this periods a_endo will be yesterday's end of period assets
                                                  # So we need the values of V today on the a_grid, this will be tomorrow's endofprd Vs 
                                                  # and used to compute tomorrow's endofprd marginal values
                    coh_cur = coh[i_n,i_z, ia]

                    interp = (a_low <= acur <= a_high) 
                    extrap = (ja == n_a - 2) and (acur > a_endo[i_n,i_z, n_a - 1])
                    extrap_low = (ja == 0) and (acur < a_endo[i_n,i_z, 0]) # This was added evaluate acur points below bottom of segment of the endogenous grid

                    # exploit that a_grid is increasing
                    if (a_high < acur < a_endo[i_n,i_z, n_a - 1]):
                        break

                    if interp or extrap or extrap_low:
                        W0 = interpolate.interpolate_point(acur, a_low, a_high, W_low, W_high)
                        a0 = interpolate.interpolate_point(acur, a_low, a_high, ap_low, ap_high)
                        c0 = coh_cur - a0 # This is the consumption today 
                                          # Overall,
                                          # a0 is the end or period assets you would choose if you were had acur amount assets today when making your consumption saving decision
                                          # c0 is the consumption you would choose if you were had acur amount assets today when making your consumption saving decision
                                          # W0 is the value tomorrow if you were had acur amount assets today when making your consumption saving decision

                        
                        if i_n ==0: # if own to own then you receive utility from owning a home
                            
                            V0 = util_c_h(c0, 1, eis, vh1, vh2,alphah ) + W0
                        else:
                            V0 = util_c_h(c0, 0, eis, vh1, vh2,alphah ) + W0
                            
                            
                        # upper envelope, update if new is better
                        if V0 > V[i_n,i_z, ia]:
                            
                            a[i_n,i_z, ia] = a0 
                            c[i_n,i_z, ia] = c0
                            V[i_n,i_z, ia] = V0
            
    return V, c, a


def constrain_policies(constraint_rent,V,c,a_x, a_grid_exp_t0,coh,eis, vh1, vh2, alphah,W,P,kappah,a_grid_exp_t1):
    
    for i_n in range(len(V[:,0,0])): # Ownership state
        for i_z in range(len(V[0,:,0])): # productivity state
            
            if i_n == 0: # own to own
     
                borrow_more = (a_x[i_n,i_z] < a_grid_exp_t0[i_n]) # boolean to flag households who want to borrow more
                below_constraint = (a_grid_exp_t0[i_n] < -P*kappah) # boolean to flag households who are below constraint
                beyond_grid = np.logical_and(borrow_more, below_constraint) # boolean to flag intersection of above
                
                a_x[i_n,i_z][beyond_grid] = a_grid_exp_t0[i_n][beyond_grid] # force households who want are flagged to remain at their current assets
                c[i_n,i_z][beyond_grid] = coh[i_n,i_z][beyond_grid] - a_x[i_n,i_z][beyond_grid] # compute consumption
                
                V[i_n,i_z][beyond_grid]  = util_c_h(c[i_n,i_z][beyond_grid],1, eis, vh1, vh2, alphah)  + W[i_n,i_z][beyond_grid]   # compute value

            elif i_n ==2: # rent to own
                
                #cstr_index = np.digitize(-P*kappah , a_grid_exp_t1[i_n]) # find index of where the borrowing constraint lies on the end of period asset grid
                
                #W_low, W_high = W[i_n,i_z,cstr_index-1], W[i_n,i_z,cstr_index] # compute end of period values on the bounds of where the constraint lies on the segment of the end of period asset grid
                #ap_low, ap_high = a_grid_exp_t1[i_n, cstr_index-1],  a_grid_exp_t1[i_n,cstr_index] # do same for end of period asset values
                #W0 = interpolate.interpolate_point(-P*kappah, ap_low, ap_high, W_low, W_high) # interpolate to find end of period value at borrowing constraint
                W0=interpolate.interpolate_y(a_grid_exp_t1[i_n] ,np.array([-P*kappah]), W[i_n,i_z,:])
                tbc = (a_x[i_n,i_z] < -P*kappah) # tbc = to become constrained
                W[i_n,i_z][tbc] = W0 

                a_x[i_n,i_z] = np.maximum(a_x[i_n,i_z], -P*kappah)   # applying borrowing constraint to households below constraint
                c[i_n,i_z] = coh[i_n,i_z] - a_x[i_n,i_z] # compute consumption
                
                V[i_n,i_z][tbc] = util_c_h(c[i_n,i_z][tbc],0, eis, vh1, vh2, alphah)  + W[i_n,i_z][tbc] # compute value


            else: # rent to rent or own to rent
                #cstr_index = np.digitize(constraint_rent , a_grid_exp_t1[i_n]) # find index of where the borrowing constraint lies on the end of period asset grid

                #W_low, W_high = W[i_n,i_z,cstr_index-1], W[i_n,i_z,cstr_index]
                #ap_low, ap_high = a_grid_exp_t1[i_n, cstr_index-1],  a_grid_exp_t1[i_n,cstr_index]
                #W0 = interpolate.interpolate_point(constraint_rent, ap_low, ap_high, W_low, W_high)
                W0=interpolate.interpolate_y(a_grid_exp_t1[i_n] ,np.array([constraint_rent]), W[i_n,i_z,:])
                                
                tbc = (a_x[i_n,i_z] < constraint_rent)# boolean to flag households below constraint

                W[i_n,i_z][tbc] = W0 

                a_x[i_n,i_z] = np.maximum(a_x[i_n,i_z], constraint_rent)
                
                c[i_n,i_z] = coh[i_n,i_z] - a_x[i_n,i_z]
                V[i_n,i_z][tbc] = util_c_h(c[i_n,i_z][tbc],0, eis, vh1, vh2, alphah) + W[i_n,i_z][tbc]

                
   
    return V,c, a_x


#%% # utility and marginal utility functions
@njit
def util_c_h(c, h, eis, vh1, vh2,alphah): # utility function with housing 
    
    x=(c**alphah) * np.exp(h*vh1)**(1-alphah)

    sigma = 1/eis
    
    if eis == 1:
        u = np.log(x) + vh2*h   
    else:
        u = x ** (1 - sigma) / (1 - sigma)  + vh2*h 


    u=u*(c>0)-100000*(c<=0)

    return u


@njit
def marg_util_c_h(c, h, eis, vh1, vh2,alphah): # marginal utility with Housing
    
    sigma = 1/eis
    x = (c**alphah) * np.exp(h*vh1)**(1-alphah)
    
    if eis == 1:
        du = (1/x)* ( c ** (alphah-1)  *  np.exp(h*vh1)**(1-alphah))    
    else:
        du = x ** (- sigma) * ( alphah * c ** (alphah - 1)  *  np.exp(h*vh1)**(1-alphah)) 
        
    du=du*(c>0)+100000*(c<=0)    
    return du 



#%% Het outputs



def efflab(z_grid,c):
    ne=np.zeros_like(c)
    ne=z_grid[np.newaxis,:,np.newaxis]+ne
    return ne

def response_by_states(c,a): # tracking consumption and asset response across different states
    
    # Consumption
    c_hoo=np.zeros_like(c)
    c_hr = np.zeros_like(c)

    c_hoo[0]=c[0] #home owners own/own 
    c_hoo[2]=c[2] #home owners rent/own
    c_hr[1]=c[1] # own/rent
    c_hr[3]=c[3] # rent/rent

    # Assets
    a_hoo=np.zeros_like(a)
    a_hr = np.zeros_like(a)
    
    a_hoo[0]=a[0] #home owners own/own
    a_hoo[2]=a[2] #home owners rent/own

    a_hr[1]=a[1] #home renters  own/rent
    a_hr[3]=a[3] #home renters  rent/rent



    return c_hoo,c_hr,a_hoo, a_hr



def housingDemand(c): # tracking housing demand

    
    hoo=np.zeros_like(c) # share of own to own
    hr=np.zeros_like(c) # share of rent to rent
    hor=np.zeros_like(c) # share of own to rent
    hro=np.zeros_like(c) # share of rent to own
    
    #home owners d'=own/own or rent/own
    hoo[0,:,:]=1 
    hoo[2,:,:]=1

    # renters d'=own/rent or rent/rent
    hr[1,:,:]=1 
    hr[3,:,:]=1

    hor[1,:,:]=1 # own to rent
    hro[2,:,:]=1 # rent to own

    return hoo,hr,hor,hro

def assetDemand(a,a_grid_r,a_grid_o): # tracking asset demand

    ad=np.zeros_like(a)
    
    ad[0,:,:]=expandx(a[0,:,:],np.min(a_grid_o),np.max(a_grid_o))
    ad[1,:,:]=expandx(a[1,:,:],np.min(a_grid_r),np.max(a_grid_r))
    ad[2,:,:]=expandx(a[2,:,:],np.min(a_grid_o),np.max(a_grid_o))
    ad[3,:,:]=expandx(a[3,:,:],np.min(a_grid_r),np.max(a_grid_r))

    return ad


#%% # Useful Functions for plotting and transforming jacobians


# function to plot IRFs
def show_irfs(irfs_list, ss0 , variables, labels=[" "], ylabel=r"(dev. from ss)", T_plot=50, figsize=(18, 6)):
    

    
    if len(irfs_list) != len(labels):
        labels = [" "] * len(irfs_list)
    n_var = len(variables)
    fig, ax = plt.subplots(1, n_var, figsize=figsize, sharex=True)
    for i in range(n_var):
        # plot all irfs
        for j, irf in enumerate(irfs_list):
            if j==1:
                if variables[i] == 'pi' or variables[i] == 'i' or variables[i] == 'r':
                    
                    ax[i].plot(  100*irf[variables[i]][:30], label=labels[j] , linestyle = '--')

                else:
                        
                    ax[i].plot(  100*irf[variables[i]][:30]/ss0[variables[i]], label=labels[j] , linestyle = '--')
            else:
                
                if variables[i] == 'pi' or variables[i] == 'i' or variables[i] == 'r':
                    
                    ax[i].plot(  100*irf[variables[i]][:30], label=labels[j] )

                else:
                    ax[i].plot(  100*irf[variables[i]][:30]/ss0[variables[i]], label=labels[j] )

                
        ax[i].plot(np.zeros(30), 'k')
        ax[i].set_title(variables[i])
        ax[i].set_xlabel(r"$t$")
        if i==0:
            ax[i].set_ylabel(ylabel)
            
        ax[i].legend()
    plt.show()
    

# function to compute consumption decomposition
def plot_C_decomp(IPR,J_hh,ss0,T):
    '''

    Parameters
    ----------
    IPR : IPR dictionary
        DESCRIPTION.
    J_hh : Jacobian dictionary
        DESCRIPTION.
    ss0 : Steady State dictionary
        DESCRIPTION.
    T : Time
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    
    G = IPR # impulse response dictionary 
    
    fig5,ax = plt.subplots()

    tt=np.arange(0,T)

    yyoldminus=0*tt[0:24]
    yyoldplus=0*yyoldminus

    #dc_lin=G['C']['epsr']@drstar*100


    bcolor=['darkblue','darkgreen','grey','gold','orange','pink','cyan']

    iter=0
    for i in ['r','Div','Tax','w','hours','pr','ph']:

        yy=J_hh['C'][i]@G[i]/ss0['C']*100 # combine HH jacobian with GE inputs 


        ax.bar(tt[:24],yy[:24].clip(min=0),bottom=yyoldplus,label=i,color=bcolor[iter])
        ax.bar(tt[:24],yy[:24].clip(max=0),bottom=yyoldminus,color=bcolor[iter])


        yyoldplus=yy[:24].clip(min=0)+yyoldplus
        yyoldminus=yy[:24].clip(max=0)+yyoldminus


        iter=iter+1

    ax.plot(G['C'][:24]/ss0['C']*100, label='Total', linestyle='-', linewidth=2.5)

    ax.legend()
    ax.set_title('Consumption Decompostion')
    

def makesticky(theta,x): # see appendix D3 of micro jumps macro humps paper

    xsticky=x*0

    xsticky[:,0]=x[:,0]    
    xsticky[0,1:x.shape[1]]=(1-theta)*x[0,1:x.shape[1]]    

    for t in range(1,x.shape[0]):
        for s in range(1,x.shape[1]):

            xsticky[t,s]=theta*xsticky[t-1,s-1]+(1-theta)*x[t,s]

    return xsticky 


# Function to convert jacobians J into sticky jacobians with sticky parameter theta
def stick_jacob(J,theta):

    Jsticky=copy.deepcopy(J)

    for i in J.outputs:

        for j in J.inputs:

            x=J[i][j]

            xsticky=makesticky(theta,x)
            Jsticky[i][j]=xsticky

    return Jsticky



def normx(x,minx,maxx): # normalise x to be between 0 and 1
    xout=(x-minx)/(maxx-minx)
    return xout


def expandx(x,minx,maxx): # expand x from 0 to 1 to minx to maxx
    xout=x*(maxx-minx)+minx
    return xout



#%% Stages


# productivity transition
prod_stage = ExogenousMaker(markov_name='z_markov', index=1, name='prod')

# taste shock function --> output is Probability of transition based on value function as input
house_choice_stage = LogitChoice(value='V', backward=['Va'], index=0, name='house_choice',
                               taste_shock_scale='taste_shock', f=util_l)

# consumption saving stage
consav_stage = Continuous1D(backward=['V', 'Va'], policy='a', f=dcegm,name='consav',hetoutputs=[efflab,housingDemand,assetDemand,response_by_states])


#%%
# Base household problem
hh = StageBlock([prod_stage, house_choice_stage, consav_stage], name='hh',
                backward_init=hh_init, hetinputs=[make_grids, labor_income])


# household problem to solve for wealth and collateral effect decomposition to change in house price
hh_wealth_effect = StageBlock([prod_stage, house_choice_stage, consav_stage], name='hh_wealth',
                backward_init=hh_init, hetinputs=[make_grids, labor_income_wealth_effect])


