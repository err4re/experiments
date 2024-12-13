import numpy as np
import data_plotting
import file_management

def flux_map_linear():

    start_time = time.time()
    elapsed_time = 0

    # initialize data variables
    S = [np.full((len(currents), N_points), np.nan) for N_points in Ns]
    f = S = [np.full((len(currents), N_points), np.nan) for N_points in Ns]

    metas = [[] for center_freq in center_freqs]


    vna.set_power(power)

    yoko.source_current()
    yoko.range_current(max(abs(currents)))
    yoko.ramp_current(currents[0], blocking=True)


    for i, current in enumerate(currents):

        # set coil current on Yoko
        yoko.current(current)

        

        for j, (frequency, span, N_points) in enumerate(zip(frequencies, spans, N_pointss)):

            points_per_segment = int(N_points/nb_segments)
                    

            #first point? otherwise use data from previous sweep
            #if i == 0:
            vna.set_sweep_type('LINear')          

            ## 1. run test sweep
            f_signal, z = test_sweep(vna, trace_name, frequency, span, bw*2, N_points)
            # else:
            #     ## use previous data from previous sweep
            #     #f_signal, z = test_sweep(vna, trace_name, frequency, span, bw, N_points)
            #     f_signal, z = f[j][:,i-1], S[j][:,i-1]
                
            
            segments = generate_segments(N_points, nb_segments, f_signal, z)


            ## 2. run segmented sweep
            ## create segments based on information from other sweep
            f[j][:,i], S[j][:,i], meta = segmented_sweep(vna, trace_name, segments, average, bw)
            metas[j].append(meta)


            # save backup (overwrite old backup)
            backup = 'backup' + file_management.generate_filename()
            np.savez(backup, center_frequencies = frequencies,
                    **dict(zip([f'f{a}' for a in range(len(frequencies))], f)),
                    **dict(zip([f'S{a}' for a in range(len(frequencies))], S)),
                    **dict(zip([f'meta{a}' for a in range(len(frequencies))], metas)),
                    currents=currents, comment=comment)
            

        elapsed_time = time.time() - start_time
        print(f'elapsed time: {elapsed_time}\n')


def flux_map_segmented(yoko, vna, currents, center_freqs, spans, Ns, power, bw, average, sample):

    start_time = time.time()
    elapsed_time = 0

    # initialize data variables
    S = [np.full((len(currents), N_points), np.nan) for N_points in Ns]
    f = S = [np.full((len(currents), N_points), np.nan) for N_points in Ns]

    metas = [[] for center_freq in center_freqs]

    data = { 
        'S' : S,
        'f' : f,
        }


    vna.set_power(power)

    yoko.source_current()
    yoko.range_current(max(abs(currents)))
    yoko.ramp_current(currents[0], blocking=True)


    for i, current in enumerate(currents):

        # set coil current on Yoko
        yoko.current(current)

        

        for j, (frequency, span, N_points) in enumerate(zip(frequencies, spans, N_pointss)):

            points_per_segment = int(N_points/nb_segments)
                    

            #first point? otherwise use data from previous sweep
            #if i == 0:
            vna.set_sweep_type('LINear')          

            ## 1. run test sweep
            f_signal, z = test_sweep(vna, trace_name, frequency, span, bw*2, N_points)
            # else:
            #     ## use previous data from previous sweep
            #     #f_signal, z = test_sweep(vna, trace_name, frequency, span, bw, N_points)
            #     f_signal, z = f[j][:,i-1], S[j][:,i-1]
                
            
            segments = generate_segments(N_points, nb_segments, f_signal, z)


            ## 2. run segmented sweep
            ## create segments based on information from other sweep
            f[j][:,i], S[j][:,i], meta = segmented_sweep(vna, trace_name, segments, average, bw)
            metas[j].append(meta)


            # save backup (overwrite old backup)
            backup = 'backup' + file_management.generate_filename(sample)
            np.savez(backup, center_frequencies = frequencies,
                    **dict(zip([f'f{a}' for a in range(len(frequencies))], f)),
                    **dict(zip([f'S{a}' for a in range(len(frequencies))], S)),
                    **dict(zip([f'meta{a}' for a in range(len(frequencies))], metas)),
                    currents=currents, comment=comment)
            

        elapsed_time = time.time() - start_time
        print(f'elapsed time: {elapsed_time}\n')