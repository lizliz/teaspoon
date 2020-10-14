
def DynamicSystems(system, dynamic_state = None, L = None, fs = None, 
                    SampleSize = None, parameters = None, 
                    InitialConditions = None, UserGuide = False):
    """This function provides a library of dynamical system models to simulate with the time series as the output. 
    
    Args:
        system (string): either 'periodic' or 'chaotic'.
        
    Other Parameters:
        dynamic_state (Optional[string]): either 'periodic' or 'chaotic'.
        L (Optional[int]): amount of time to solve simulation for.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[array]): dynamic system parameters.
        InitialConditions (Optional[array]): initial conditions for simulation.
        UserGuide (Optional[bool]): 
    
    Returns:
        array: Array of the time indices as t and the simulation time series ts from the simulation for all dimensions of dynamcal system (e.g. 3 for Lorenz).
    """
    
    #needed packages
    import numpy as np
    import autonomous_dissipative_flows
    import maps
    import periodic_functions
    import noise_models
    import medical_data
    import delayed_flows
    import conservative_flows
    import driven_dissipative_flows
    
    t, ts =  [None], [None] #primes the time and time series datas.
    
    if dynamic_state == None and parameters == None:
        print('Error: need to provide dynamic state or desired system parameters.')
        print('Defaulting to periodic dynamic state.')
        dynamic_state = 'periodic'
    
    
    list_of_autonomous_dissipative_flows = ['chua', 'lorenz', 'rossler', 'coupled_lorenz_rossler',
                           'coupled_rossler_rossler', 'double_pendulum', 'diffusionless_lorenz_attractor',
                           'complex_butterfly', 'chens_system', 'hadley_circulation', 'ACT_attractor', 
                           'rabinovich_frabrikant_attractor', 'linear_feedback_rigid_body_motion_system', 
                           'moore_spiegel_oscillator', 'thomas_cyclically_symmetric_attractor',
                           'halvorsens_cyclically_symmetric_attractor', 'burke_shaw_attractor',
                           'rucklidge_attractor', 'WINDMI', 'simplest_quadratic_chaotic_flow', 
                           'simplest_cubic_chaotic_flow', 'simplest_piecewise_linear_chaotic_flow',
                           'double_scroll']
    
    list_of_periodic_functions = ['sine', 'incommensurate_sine']
    
    list_of_maps = ['logistic_map', 'henon_map', 'sine_map', 'tent_map', 'linear_congruential_generator_map', 
                        'rickers_population_map', 'gauss_map', 'cusp_map', 'pinchers_map', 'sine_circle_map',
                        'lozi_map', 'delayed_logstic_map', 'tinkerbell_map',  'burgers_map', 'holmes_cubic_map',
                        'kaplan_yorke_map']
    
    list_of_noise_models = ['gaussian_noise','uniform_noise', 
                                'rayleigh_noise', 'exponential_noise']
    
    
    list_of_medical_data = ['ECG', 'EEG']
    
    list_of_delayed_flows = ['mackey_glass']
    
    list_of_driven_dissipative_flows = ['driven_pendulum', 'driven_can_der_pol_oscillator', 'shaw_van_der_pol_oscillator',
                                        'forced_brusselator', 'ueda_oscillator', 'duffings_two_well_oscillator',
                                        'duffing_van_der_pol_oscillator', 'rayleigh_duffing_oscillator']
    
    list_of_conservative_flows = ['simplest_driven_chaotic_flow', 'nose_hoover_oscillator',
                                  'labyrinth_chaos', 'henon_heiles_system']
    
    if UserGuide == True: #prints the user guide and all available dynamical systems.
        import UserGuide
        UserGuide.UserGuide(list_of_autonomous_dissipative_flows,
                            list_of_conservative_flows,
                            list_of_periodic_functions,
                            list_of_maps,
                            list_of_noise_models,
                            list_of_medical_data,
                            list_of_delayed_flows,
                            list_of_driven_dissipative_flows)
    
    all_systems = []
    for systems in [list_of_autonomous_dissipative_flows,
                    list_of_conservative_flows,
                    list_of_periodic_functions,
                    list_of_maps,
                    list_of_noise_models,
                    list_of_medical_data,
                    list_of_delayed_flows,
                    list_of_driven_dissipative_flows]:
        for sys in systems:
            all_systems.append(sys)

        
# In[ ]: AUTONOMOUS DISSIPATIVE FLOWS
    if system in list_of_autonomous_dissipative_flows:
        t, ts = autonomous_dissipative_flows.autonomous_dissipative_flows(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)

# In[ ]: MAPS
    if system in list_of_maps:
        t, ts = maps.maps(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)

# In[ ]: PERIODIC
    if system in list_of_periodic_functions:
        t, ts = periodic_functions.periodic_functions(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)
                
# In[ ]: NOISE
    if system in list_of_noise_models:
        t, ts = noise_models.noise_models(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)

# In[ ]: MEDICAL
    if system in list_of_medical_data:
        t, ts = medical_data.medical_data(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)
    
# In[ ]: DDE
    if system in list_of_delayed_flows:
        t, ts = delayed_flows.delayed_flows(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)
         
# In[ ]: driven_dissipative_flows
    if system in list_of_driven_dissipative_flows:
        t, ts = driven_dissipative_flows.driven_dissipative_flows(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)
        
# In[ ]: conservative_flows
    if system in list_of_conservative_flows:
        t, ts = conservative_flows.conservative_flows(system, dynamic_state, L, fs, 
                                             SampleSize, parameters, InitialConditions)

        
# In[ ]:
    
    if t[0] == None: 
        print(system, 'model or data requested does not exist.')
        t = np.nan
        ts = [np.nan]
        
    return t, ts

                
# In[ ]:
    
if __name__ == "__main__":
    
    
    import DynSysLib as DSL
    system = 'rossler' #define the system of interest
    dynamic_state = 'periodic' #set the dynamic state
    t, ts = DSL.DynamicSystems(system, dynamic_state, UserGuide = True)

    
    
    #this plots the resulting time series
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    TextSize = 15
    plt.figure(figsize = (14,4))
    gs = gridspec.GridSpec(1,2) 
    
    ax = plt.subplot(gs[0, 0])
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.ylabel(r'$x(t)$', size = TextSize)
    plt.xlabel(r'$t$', size = TextSize)
    plt.plot(t,ts[0], 'k')
    
    if len(ts) > 1:
        ax = plt.subplot(gs[0, 1])
        plt.plot(ts[0], ts[1],'k.', markersize = 2)
        plt.plot(ts[0], ts[1],'k', alpha = 0.25)
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.xlabel(r'$x(t)$', size = TextSize)
        plt.ylabel(r'$y(t)$', size = TextSize)
        
    plt.show()
    