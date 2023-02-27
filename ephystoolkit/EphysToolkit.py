# class imports
import re
import math
import json
import warnings

import numpy as np
import scipy.io
import pandas as pd
from scipy import stats

from pathexplorer.PathExplorer import path_explorer


class ephys_toolkit:

    def __init__(self):
        self.SAMPLING_RATE = 20000

    def _bin_events(self, bin_size, events):
        self.frames = bin_size ** -1
        self.numerator = self.SAMPLING_RATE / self.frames

        return events / self.numerator

    def _minmax_norm(self, array):
        """
        Normalize an array with minmax normalization.
        """
        return (array - array.min()) / (array.max() - array.min())

    def _zscore_norm(self, array):
        """
        Normalize an array with z score normalization.
        """
        return stats.zscore(array)

    def _average_response(self, array, stim_reps):
        """
        Return the average frequency of a unit's response
        across stimulus repeats.
        """
        return (array / stim_reps) * self.frames
    
    def static_grating(
            self,
            windowSizeX_Pixel, windowSizeY_Pixel, # size of the stimulus window in pixels
            windowSizeX_Visual, windowSizeY_Visual, # size of the stimulus window in degrees
            spatialFrequency, # spatial frequency in cpd
            ang, # orientation angle of the grating in degrees
            phi, # phase of the grating in degrees
            diameter = None # dimater of the circular patch in degrees
    ):    
        """
        Generate a matrix of pixel intensities   
        representing a static grating stimulus
        with the given parameters.
        
        Args:
        
        - windowSizeX_Pixel: Horizontal size of the stimulus screen in pixels.
        - windowSizeY_Pixel: Vertical size of the stimulus screen in pixels.
        - windowSizeX_Visual: Horizontal size of the stimulus screen in degrees.
        - windowSizeY_Visual: Vertical size of the stimulus screen in degrees.
        - spatialFrequency: Spatial frequency in cycles per degre (cpd).
        - ang: Orientation angle of the grating in degrees.
        - phi: Phase of the grating in degrees
        - diameter: Dimater of the circular patch in degrees. 
        """
        
        if ((windowSizeX_Pixel <=10)
            or (windowSizeX_Pixel%2 != 0)
        ):
            print('windowSizeX_Pixel must be greater than ten and even')

        if ((windowSizeY_Pixel <=10)
            or (windowSizeY_Pixel%2 != 0)
        ):
            print('windowSizeY_Pixel must be greater than ten and even')

        if (math.floor(windowSizeX_Pixel/windowSizeY_Pixel*1000) 
            != math.floor(windowSizeX_Visual/windowSizeY_Visual*1000)
        ):
            print('The ratio of X and Y for Pixel and Visual are different!');

        if spatialFrequency < 0:
            error('spatialFrequency less than zero')

        # x, y 
        x_center = windowSizeX_Pixel/2
        y_center = windowSizeY_Pixel/2

        x_range = np.arange(-x_center, x_center,1)
        y_range = np.arange(-y_center, y_center,1)
        x, y = np.meshgrid(x_range,y_range)

        # theta
        theta = -np.radians(ang);
        xyTheta = y * np.cos(theta) - x * np.sin(theta);

        # Spatial Frenquency
        degreePerPixel = windowSizeX_Visual / windowSizeX_Pixel
        sf = spatialFrequency  * degreePerPixel # cycles / pixel
        sw = 2 * np.pi * sf # radian / pixel

        # contrast
        pixelIntensity = (np.sin(sw * xyTheta - phi)) # range: -1 to 1

        # round image
        if diameter == None:
            pass
        else:
            diameter = diameter / degreePerPixel
            r = diameter/2;
            c_mask = ( (x**2+y**2) <= r**2 )
            pixelIntensity = pixelIntensity*c_mask 

        return pixelIntensity

    def drifting_grating(
            self,
            windowSizeX_Pixel, windowSizeY_Pixel, # size of the stimulus window in pixels
            windowSizeX_Visual, windowSizeY_Visual, # size of the stimulus window in degrees
            tf,  # Temporal frequency - pass as an integer or floating point value
            dt,  # Time step value - pass as a floating point value
            t,  # Total duration of the stimulus - pass as an int or float of the appropriate time unit
            spatialFrequency, # spatial frequency in cpd
            ang, # orientation angle of the grating in degrees
            diameter = None # dimater of the circular patch in degrees
    ):
        """
        Returns a list of matricies representing
        frames of a drifitng grating stimulus.
        
        Args:
        - windowSizeX_Pixel: Horizontal size of the stimulus screen in pixels.
        - windowSizeY_Pixel: Vertical size of the stimulus screen in pixels.
        - windowSizeX_Visual: Horizontal size of the stimulus screen in degrees.
        - windowSizeY_Visual: Vertical size of the stimulus screen in degrees.    
        - tf: Temporal frequency - pass as an integer or floating point value.
        - dt: Time step value - pass as a floating point value.
        - t: Total duration of the stimulus - pass as an int or float of the appropriate time unit.
        - spatialFrequency: Spatial frequency in cycles per degre (cpd).
        - ang: Orientation angle of the grating in degrees.
        - diameter: Dimater of the circular patch in degrees. 
        
        """

        tensor = []
        params = []
        deg_step = dt * 360 * tf

        d = np.arange(dt, t, dt)

        phi = 0
        for x in d:
            m = self.static_grating(
                windowSizeX_Pixel, windowSizeY_Pixel, 
                windowSizeX_Visual, windowSizeY_Visual, 
                spatialFrequency,
                ang,
                phi,
                diameter = diameter
            ) 
            tensor.append(m)
            params.append([sf, tf, ori, phase])
            phi += deg_step

        return tensor, params

    def _discrete_radius(self, dim=tuple, radius=int):
        x = np.linspace(-1, 1, dim[1])
        y = np.linspace(-1, 1, dim[0])
        

        m = []
        for i0 in range(len(x)):
            row = []
            for i1 in range(len(y)):
                if ((x[i0] ** 2 + y[i1] ** 2)
                        < ((radius / 360) ** 2)):
                    row.append(1)
                else:
                    row.append(0)
            m.append(row)

        return np.array(m).T

    def _gaussian_radius(self, dim=tuple, radius=int):
        x, y = np.meshgrid(
            np.linspace(-1, 1, dim[0]),
            np.linspace(-1, 1, dim[1])
        )
        d = np.sqrt(x * x + y * y)
        sigma, mu = radius / 360, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

        return g

    def spike_sorting_metrics(self, file_path):
        """
        Returns a dataframe with the spike sorting metrics
        of a given recording section.
        
        Args:
        
        - file_path: Path to the spike sorting metrics file.
        """

        with open(file_path, 'r') as sorting_file:
            sorting_info = json.load(sorting_file)

        spike_sorting_data = [
            # [cluster['label'] for cluster in sorting_info['clusters']],
            [cluster['metrics']['isolation'] for cluster in sorting_info['clusters']],
            [cluster['metrics']['noise_overlap'] for cluster in sorting_info['clusters']]
        ]

        ss_df = pd.DataFrame(np.array(spike_sorting_data).T)
        ss_df.columns = [
            # 'cluster', 
            'isolation', 'noise_overlap'
        ]

        return ss_df

    # functions to make concatenated across trial and non concatenated across trial rasters
    def _concatenated_raster(self, stims, spikes, thresh=tuple):
        if thresh == tuple:
            r = np.array([spikes - st for st in stims])
            raster = np.concatenate(r)
        else:
            r = np.array([spikes - st for st in stims])
            ti = np.where(np.logical_and(r <= thresh[1], r >= thresh[0]))
            raster = r[ti]
        return raster

    def _unconcatenated_raster(self, stims, spikes, thresh=tuple):
        if thresh == tuple:
            rasters = np.array([spikes - st for st in stims])
        else:
            rasters = []
            for i, st in enumerate(stims):  # enumerate to make an initial array then vstack
                unthreshed = spikes - st
                i = np.where(np.logical_and(unthreshed <= thresh[1], unthreshed >= thresh[0]))
                rasters.append(list(unthreshed[i]))
        return rasters

    def raster(
            self,
            stims,  # Array of stimulus onset times
            spikes,  # Array of spike onset times
            thresh=tuple,  # Bounding threshold around the stimulus onset at t = 0 - pass as a tuple
            concatenate=True  # Concatenate rasters across trials - pass False to return unconcatenated rasters
    ):
        """
        Returns an array representing a raster of spike times centered 
        around the onset times of a given stimulus.
        
        Args:
        
        - stims: Array of stimulus onset times.
        - spikes: Array of spike onset times.
        - thresh: Bounding threshold around the stimulus onset at t = 0 - pass as a tuple.
        - concatenate: Concatenate rasters across trials; pass False to return unconcatenated rasters.
        
        """

        if concatenate:
            return self._concatenated_raster(stims, spikes, thresh)
        else:
            return self._unconcatenated_raster(stims, spikes, thresh)

    def avg_across_param(
            self,
            pop_resp, # this df must have cluster ids as columns
            col_param: str, # parameter whose values are to be shown in the columns
            avg_param: list, # parameters to average across
    ):
        """
        Returns a dataframe showing responses as a function one parameter while averaging
        over the other parameters.
        
        Args:
        - pop_resp: A dataframe of the population response in which the columns are units.
        - col_param: The parameter to show in the columns.
        - avg_param: The parameter(s) to average across.
        """

        # groupby the parameter of interest
        gb = pop_resp.groupby(col_param).agg(list)

        # get an array of the column values and cluster ids
        col_values = gb.index.values
        units = [x for x in gb.columns.values if re.search(r'\d', x)]

        # initialize dictionary for cross parameter averaged data
        data = {'cluster': np.array([])}
        for col in col_values:
            data[col] = np.array([])

        # create the cluster id column
        for unit_id in units:
            unit_ids = np.array([unit_id]*500)
            data['cluster'] = np.concatenate((data['cluster'], unit_ids), axis = 0)

        # get the number of parameter combos and length of response
        param_combos = 1
        for param in avg_param:
            param_combos*=len(pop_resp[param].unique())
        resp_len = len(pop_resp.loc[pop_resp.stimulus_condition == 1])

         # create the firing rate value columns
        try:
            for col in col_values:
                for unit_id in units:
                    cc_avg = np.array(gb[unit_id][col]).reshape(param_combos,resp_len).mean(0)
                    data[col] = np.concatenate((data[col], cc_avg), axis = 0)
            return pd.DataFrame(data)
        except ValueError:
                        err_msg = """
                        Failed to reshape response array. 
                        Make sure col_param is not repeated
                        in avg_param.
                        """
                        print(err_msg)

class load_experiment(ephys_toolkit):
    """
    Create an experiment object for a given recording block.
    Takes the spike data file path as the first argument and the
    stimulus data file path as the second argument. Initializing
    an experiment object generates some important class attributes:
    
    .stim_data: A pandas dataframe with the stimulus data.
    
    .spike_data: A dictionary object with the spiking data
                 of all the identified clusters.
                 
    Args:
    
    - spikefile: Path to the spike data file.
    - stimfile: Path to the stimulus data file.
    """

    def __init__(self, spikefile, stimfile, logfile):
        ephys_toolkit.__init__(self)

        spikes_mat = scipy.io.loadmat(spikefile)
        stims_mat = scipy.io.loadmat(stimfile)

        self.spikes = spikes_mat['Data'][0]
        self.stims = stims_mat['StimulusData'][0][0]
        self.logfile = logfile
        self._init_stim_data()
        self._init_spike_data()
        self.set_time_unit()

        self.parameters_matched = False

        self.stim_conditions = self.stim_data[
            'stim_condition_ids'
        ].unique()
        
        if self.logfile != None:
            self._get_conditions_from_log()
        else:
            r = r'([A-Z]{1,2}_M\d{1,}_Section_\d{1,}_BLK\d{1,})'
            experiment_id = re.search(r, spikefile).group(1)
            warn_text = f"""
            No stimulus log file found for experiment: {experiment_id}.
            If this experiment has more that 255 stimulus conditions, the stimulus 
            condition IDs will be incorrect in the self.stim_data attribute.
            """
            warnings.warn(warn_text, stacklevel = 4)

    # Generates the .stim_data attribute.
    def _init_stim_data(self):

        stims_starts = self.stims[0]
        stims_stops = self.stims[1]
        stims_condition_ids = self.stims[2]

        stim_data = {
            'stim_start_indicies': stims_starts[0],
            'stim_stop_indicies': stims_stops[0],
            'stim_condition_ids': stims_condition_ids[0]
        }

        self.stim_data = pd.DataFrame(stim_data)
        """
        A pandas dataframe with the stimulus data.
        """

    # Generates the .spike_data attribute.
    def _init_spike_data(self):

        self.spike_data = [
            {
                # 'cluster_id':unit[0][0][0],
                'channel_id': unit[1][0][0],
                'spike_index': unit[2][0],
                'spike_time': unit[3][0]
            }
            for unit in
            self.spikes
        ]
        """
        A dictionary object with the spiking data.
        """
        
    def _get_conditions_from_log(self):
        
        with open(self.logfile, 'r') as f:
            stimlog = f.readlines()

        stimlog = stimlog[1:-2]
        
        # extract conditions for log
        r0 = r'[Cc]ondition#: *(\d{1,})'
        r1 = r'[Cc]onditions:(\d{1,} +...*)'
        condition_ids = []
        
        if re.search(r0, stimlog[1]): # if the experiment is gratings or checkerboard
            for line in stimlog:
                c_id = re.search(r0, line).group(1)
                condition_ids.append(int(c_id))
                
            try:  
                # replace conditions in experiment with conditions from log
                self.stim_data.stim_condition_ids = condition_ids
                self.stim_conditions = self.stim_data.stim_condition_ids.unique()
            except ValueError:
                pass
            
        elif re.search(r1, stimlog[1]): # if the experiment is natural images
            condition_ids = []
            for x in stimlog:
                cindex = x.index(':')+1 # index where condition numbers begin
                numbers = x[cindex:].strip() # remove whitespace at the end
                numbers = numbers.replace('  ', ' ') # remove double spaces
                numbers = numbers.split(' ') # split into a list of conditon numbers
                numbers = [int(x) for x in numbers] # format str into int
                condition_ids += numbers #collect condition ids

            ## This horrible block of code is here because the stimData matlab
            ## file is missing the data the last two stimulus repeats
            ## so it is added manually until the stimData file generation is fixed

            # difference between consecutive start indicies
            startindex_diff = (self.stim_data['stim_start_indicies'].values[-1] 
                             - self.stim_data['stim_start_indicies'].values[-2])

            # difference between consecutive stop indicies
            stopindex_diff = (self.stim_data['stim_stop_indicies'].values[-1] 
                             - self.stim_data['stim_stop_indicies'].values[-2])

            # difference between consecutive start times
            starttime_diff = (self.stim_data['stim_start_times'].values[-1] 
                             - self.stim_data['stim_start_times'].values[-2])

            # difference between consecutive stop times
            stoptime_diff = (self.stim_data['stim_stop_times'].values[-1] 
                             - self.stim_data['stim_stop_times'].values[-2])

            # adding two additional start indicies 
            start_ind_fixed = list(self.stim_data['stim_start_indicies'].values)
            start_ind_fixed = start_ind_fixed + [start_ind_fixed[-1]-+startindex_diff, 
                                                 start_ind_fixed[-1]+2*startindex_diff]
            # adding two additional stop indicies 
            stop_ind_fixed = list(self.stim_data['stim_stop_indicies'].values)
            stop_ind_fixed = stop_ind_fixed + [stop_ind_fixed[-1]+stopindex_diff, 
                                                 stop_ind_fixed[-1]+2*stopindex_diff]

            # adding two additional start times 
            start_time_fixed = list(self.stim_data['stim_start_times'].values)
            start_time_fixed = start_time_fixed + [start_time_fixed[-1]+starttime_diff, 
                                                 start_time_fixed[-1]+2*starttime_diff]
            # adding two additional stop times 
            stop_time_fixed = list(self.stim_data['stim_stop_times'].values)
            stop_time_fixed = stop_time_fixed + [stop_time_fixed[-1]+stoptime_diff, 
                                                 stop_time_fixed[-1]+2*stoptime_diff]

            # fixed stim data dictionary
            stim_data_fixed = {
                'stim_condition_ids': condition_ids,
                'stim_start_indicies': start_ind_fixed,
                'stim_stop_indicies': stop_ind_fixed,
                'stim_start_times': start_time_fixed,
                'stim_stop_times': stop_time_fixed
            }
            
            self.stim_data = pd.DataFrame(stim_data_fixed)
            self.stim_conditions = self.stim_data.stim_condition_ids.unique()
            
    def set_time_unit(self, bin_size=0.001):
        """
        Change the time unit of the relative spike times.
        Give a bin size relative to 1 second.
        IE: If you want a 1 ms bin size, enter 0.001;
        if you want a 10 ms bin size, enter 0.01 etc.
        
        Args:
        
        - bin_size: Time unit given relative to 1 second. The default unit is 1ms/0.001s.
        """

        self.stim_data['stim_start_times'] = self._bin_events(bin_size,
                                                              self.stim_data['stim_start_indicies'].values)

        self.stim_data['stim_stop_times'] = self._bin_events(bin_size,
                                                             self.stim_data['stim_stop_indicies'].values)

        self.stim_data = self.stim_data[
            ['stim_condition_ids',
             'stim_start_indicies',
             'stim_stop_indicies',
             'stim_start_times',
             'stim_stop_times']
        ]

        for unit in self.spike_data:
            unit.update({
                'rel_spike_time': self._bin_events(bin_size,
                                                   unit['spike_index'])
            })

    def condition_times(self, condition):
        """
        Returns a dictionary object of the start times and
        stop times of a particular stimulus condition.
        
        Args:
        
        -condition: Condition id for the chosen stimulus condition.
        """

        condition_starts = self.stim_data.groupby(
            self.stim_data.stim_condition_ids
        ).get_group(condition)['stim_start_times'].values

        condition_stops = self.stim_data.groupby(
            self.stim_data.stim_condition_ids
        ).get_group(condition)['stim_stop_times'].values

        return {
            'start': condition_starts,
            'stop': condition_stops
        }

    def match_condition_parameters(self, params_file):
        """
        Takes a parameters file and alters the .stim_data dataframe
        to match stimulus condition parameters to their corresponding 
        condition id.
        
        Args:
        
        - params_file: Path to the stimulus parameters file.
        """
        # 

        # regex to get the parameter names + values
        name_identifier = r'Var\d{1,}Val'
        val_identifier = r'var\d{1,}value'

        # load the parameter file and extract the keys
        params = scipy.io.loadmat(params_file)
        param_keys = list(params.keys())
        
        if 'fnames' in param_keys: # if experiment is natural images
            
            # get the stim file names
            stim_file_names = [
                name[0][0] for 
                name in scipy.io.loadmat(params_file)['fnames']
            ]
            
            # make a dictionary for the parameter mapping
            ni_filemap = {
                'condition': [],
                'size': [],
                'filter': [],
                'file': []
            }
            
            # match condition to parameters
            freg = r'^[A-Z]{2,3}'
            for i in self.stim_conditions:
                if i%2 == 0: # if condition is even
                    
                    #append condition & size
                    ni_filemap['condition'].append(i)
                    ni_filemap['size'].append(60)
                    
                    #append filename
                    filename = stim_file_names[int(i/2)-1]
                    ni_filemap['file'].append(filename) 
                    
                    #append filtering condition
                    filt = re.search(freg, filename).group(0)
                    ni_filemap['filter'].append(filt)
            
                else: # if condition is odd
                    #append condition & size
                    ni_filemap['condition'].append(i)
                    ni_filemap['size'].append(30)
                    
                    #append filename
                    filename = stim_file_names[int((i+1)/2)-1]
                    ni_filemap['file'].append(filename)
                    
                    #append filtering condition
                    filt = re.search(freg, filename).group(0)
                    ni_filemap['filter'].append(filt)
                    
            self.parameter_map = pd.DataFrame(ni_filemap) 
            df = self.stim_data
            
            # get the parameters to insert into the original dataframe
            insert = []
            for cond in df.stim_condition_ids.values:
                arr = self.parameter_map.loc[self.parameter_map.condition == cond].values[0]
                insert.append(arr)
            insert = pd.DataFrame(insert, columns=self.parameter_map.columns.values)

            # insert the parameter values into the original dataframe
            df1 = df.join(insert)

            # reset the stim_data attribute
            self.stim_data = df1[
                list([df.columns.values[0]])
                + list(self.parameter_map.columns.values[1:])
                + list(
                    df.columns.values[1:]
                )
                ]
        
        else: # if experiment is gratings or checkerboard
            # use regex to get the param names and values
            name_keys = [key for key in param_keys if re.search(name_identifier, key)]
            val_keys = [key for key in param_keys if re.search(val_identifier, key)]

            # get the condition numbers
            condition_range = len(params[val_keys[0]][0])
            condition_ids = [i + 1 for i in range(condition_range)]

            # map conditions to parameter values
            parameter_map = {'condition': condition_ids}
            for i in range(len(name_keys)):
                parameter_name = str(params[name_keys[i]][0])
                parameter_values = np.array(params[val_keys[i]][0])

                parameter_map[parameter_name] = parameter_values

                # parameter dataframe + the original stim_data dataframe
            self.parameter_map = pd.DataFrame(parameter_map)
            df = self.stim_data

            # get the parameters to insert into the original dataframe
            insert = []
            for cond in df.stim_condition_ids.values:
                arr = self.parameter_map.loc[self.parameter_map.condition == cond].values[0]
                insert.append(arr)
            insert = pd.DataFrame(insert, columns=self.parameter_map.columns.values)

            # insert the parameter values into the original dataframe
            df1 = df.join(insert)

            # reset the stim_data attribute
            self.stim_data = df1[
                list([df.columns.values[0]])
                + list(self.parameter_map.columns.values[1:])
                + list(
                    df.columns.values[1:]
                )
                ]

        self.parameters_matched = True

    def _pr_unitcols(
            self,
            include_units,
            stim_condition=None, 
            columns='cluster_id',
            thresh=None,
            norm=None 
    ):

        if thresh is None:
            return "Please provide a response boundary threshhold as a tuple (ex: (-50,500))"
        else:
            thresh_min = thresh[0] - 0.5
            thresh_max = thresh[1] + 0.5

        if stim_condition is 'all':
            stim_condition = self.stim_conditions

        population = {}
        cond_col = []
        param_col = []

        for condition in stim_condition:
            condition_start_times = self.condition_times(condition)['start']

            for i, cluster_id in enumerate(include_units):
                if type(cluster_id) == np.ndarray:
                    unit_spike_times = cluster_id
                    cluster_id = f"n{i}"
                    
                else:
                    unit_spike_times = self.spike_data[cluster_id]['rel_spike_time']

                # Raster
                x = self.raster(
                    condition_start_times, 
                    unit_spike_times,
                    thresh=thresh)

                # Raster to histogram 
                h, bins = np.histogram(
                    x, 
                    bins=np.arange(thresh_min,thresh_max,1)
                )

                # Nonresponsive unit histogram
                if h.min() == 0 and h.max() == 0:
                    h = np.zeros(len(h))

                # Normalize the response
                else:
                    pass
                    norm_method = {
                        'minmax': self._minmax_norm(h),
                        'zscore': self._zscore_norm(h),
                        'average': self._average_response(h, len(condition_start_times))
                    }

                    if norm is None: continue

                    elif (norm is not None
                          and norm in list(norm_method.keys())):
                        h = norm_method[norm]
                    else:
                        raise _UnrecognizedNorm(
                            f"Normalization method is not recognized,\
                            please choose from the following:\
                            {list(norm_method.keys())}"
                        )

                # fill the population dictionary
                if cluster_id not in population:
                    population[cluster_id] = h
                else:
                    population[cluster_id] = np.concatenate(
                        (population[cluster_id], h)
                    )

            # append condition ids to the conditions column
            cond_col += [int(condition)]*len(h)

            # append condition parameters
            if self.parameters_matched:
                params = list(
                    self.parameter_map.loc[self.parameter_map.condition == condition].values[0][1:]
                )
                param_col += [params] * len(h)

        # Rearrange the dataframe
        if self.parameters_matched:

            # Make the population & stimulus parameters dataframes
            population = pd.DataFrame(population)
            param_col = pd.DataFrame(
                param_col, columns=self.parameter_map.columns.values[1:]
            )

            # Retrieve their column labels
            pcol = list(population.columns.values)
            pacol = list(param_col.columns.values)

            # Add the condition id column & the stimulus parameters columns
            population['stimulus_condition'] = cond_col
            population = population.join(param_col)

            # Rearrange column order
            new_columns = ['stimulus_condition'] + pacol + pcol
            population = population[new_columns]

        else:
            # Make the population dataframe
            population = pd.DataFrame(population)

            # Retrieve its column labels
            pcol = list(population.columns.values)

            # Add the condition id column
            population['stimulus_condition'] = cond_col

            # Rearrange column order
            population = population[['stimulus_condition']+pcol]

        return population       

    def _pr_stimcols(
            self,
            include_units, 
            stim_condition=None, 
            columns='cluster_id',
            thresh=None, 
            norm=None
    ):

        if thresh is None:
            return "Please provide a response boundary threshhold as a tuple (ex: (-50,500))"
        else:
            thresh_min = thresh[0] - 0.5
            thresh_max = thresh[1] + 0.5

        if stim_condition is 'all':
            stim_condition = np.sort(self.stim_conditions)

        population = {}
        unit_col = []

        for cluster_id in include_units:
            unit_spike_times = self.spike_data[cluster_id]['rel_spike_time']

            for condition in stim_condition:
                condition_start_times = self.condition_times(condition)['start']

                # Raster
                x = self.raster(
                    condition_start_times, 
                    unit_spike_times,
                    thresh=thresh)

                # Raster to histogram 
                h, bins = np.histogram(
                    x, 
                    bins=np.arange(thresh_min,thresh_max,1)
                )

                # Nonresponsive unit histogram
                if h.min() == 0 and h.max() == 0:
                    h = np.zeros(len(h))

                # Normalize the response
                else:
                    pass
                    norm_method = {
                        'minmax': self._minmax_norm(h),
                        'zscore': self._zscore_norm(h),
                        'average': self._average_response(h, len(condition_start_times))
                    }

                    if norm is None: continue

                    elif (norm is not None
                          and norm in list(norm_method.keys())):
                        h = norm_method[norm]
                    else:
                        raise _UnrecognizedNorm(
                            f"Normalization method is not recognized,\
                            please choose from the following:\
                            {list(norm_method.keys())}"
                        )

                # fill the population dictionary
                if condition not in population:
                    population[int(condition)] = h
                else:
                    population[condition] = np.concatenate(
                        (population[int(condition)], h)
                    )

            # append condition ids to the conditions column
            unit_col += [cluster_id]*len(h)

        # Rearrange the dataframe
        population = pd.DataFrame(population)
        pcol = list(population.columns.values)
        population['cluster_id'] = unit_col
        population = population[['cluster_id']+pcol]

        return population       

    def population_response(
            self,
            include_units,  # Units to include in the dataframe
            stim_condition=None,  # Stimulus condition(s) to include in the dataframe
            columns='cluster_id',  # Set column label arrangement
            thresh=None,  # Bounding threshold around the stimulus onset at t = 0 - pass as a tuple
            norm=None  # Normalization method - choose from: 'minmax', 'zscore', or 'average'
    ):
        """
        Returns a dataframe of the population response PSTH.
        By default, each column label represents the
        included units. A single column identifies
        the stimulus condition at each row. Each row is the
        average response at each time step.

        Setting columns = "stimulus_condition" will
        return a data frame where each column label
        represents a stimulus condition. A single
        column identifes the included unit at each row.
        Each row is the average response at each time step.

        Args:

        - include_units: Units to include in the dataframe - pass as a 1d array or list like object.
        - stim_condition: Stimulus condition(s) to include in the dataframe - 
          pass a list of condition ids or 'all' to include all conditions.
        - columns: Set column label arrangement - pass either 'cluster_id' or 'stimulus_condition'. 
          Default argument is 'stimulus_condition'.
        - thresh: Bounding threshold around the stimulus onset at t = 0 - pass as a tuple.
        - norm: Normalization method - pass either 'minmax', 'zscore', or 'average'.
        """

        # Case checks:
        if stim_condition is None:
            raise _NoStimulusCondition()
        elif stim_condition is 'all': pass
        elif set(stim_condition).issubset(set(self.stim_conditions)): pass
        else:
            raise _UnrecognizedStimulusCondition()

        col_formats = ['cluster_id', 'stimulus_condition'] 
        if columns not in col_formats:
            raise _UnrecognizedColumnsInput(list(columns_dic.keys()))

        elif columns == 'cluster_id':
            population = self._pr_unitcols(
            include_units,
            stim_condition=stim_condition,
            columns=columns,
            thresh=thresh,
            norm=norm)

        elif columns == 'stimulus_condition':
            population = self._pr_stimcols(
            include_units,
            stim_condition=stim_condition,
            columns=columns,
            thresh=thresh,
            norm=norm)

        return population
                    
    def _stim_frame_map(self, dim, radius, edge):
        frame_map = {}
        for i in range(len(self.parameter_map)):
            con = i+1
            dx = self.parameter_map.iloc[i]

            frame = self.static_grating(
                sf = float(dx['Spatial Freq']),
                ori = float(dx['Orientation']),
                ph = float(dx['Phase'])*360,
                dim = dim,
                radius = radius,
                edge = edge
            )

            frame_map[con] = frame

        self.frame_map = frame_map

    def _stim_frames(self):
        cond = self.stim_data.stim_condition_ids.values
        self.stim_frames = np.array([self.frame_map[i] for i in cond])


class load_project(ephys_toolkit):
    """
    Initialize the load_project class with a full path to the
    directory containing the project files.
    
    The .workbook attribute contains a list of dictionaries
    with the following structure:
    

      
          {
          
              'section_id': int,
              
              'spike_sorting_metrics': dataframe,
              
              'blocks': [
              
                  {'block_id': int, 'experiment', experiment object},
                  
                  ]
                  
          },
    
    Args:
    
    - project_path: Path to the directory containing the project files.
    """

    def __init__(self, project_path):
        ephys_toolkit.__init__(self)
        self.ppath = project_path
        self._init_project_workbook()

    # generate the workbook of project data
    def _init_project_workbook(self):
        explorer = path_explorer()

        # find and sort spike files
        spike_files = explorer.findext(self.ppath, '.mat', r='firings')
        spike_files.sort(key=lambda x: int(re.search(r'BLK(\d{1,})', x).group(1)))

        # find and sort stim files
        stim_files = explorer.findext(self.ppath, '.mat', r='stimulusData')
        stim_files.sort(key=lambda x: int(re.search(r'BLK(\d{1,})', x).group(1)))
        
        # find and sort log files
        log_files = explorer.findext(self.ppath, '.log')
        log_files.sort(key=lambda x: int(re.search(r'BLK(\d{1,})', x).group(1)))
        
        # count how many log files are in the data folder
        log_diff = len(spike_files)-len(log_files)
        for i in range(log_diff):
            log_files.append(None)

        # zip matching blocks
        matched_block_files = zip(spike_files, stim_files, log_files)

        # find metrics files
        metrics_files = explorer.findext(self.ppath, '.json', r='metrics_isolation')

        ################################################################################

        # compile the workbook
        self.workbook = []

        # compile spike sorting metrics first
        for metrics_file in metrics_files:
            section_parent = int(re.search(r'Section_(\d{1,})', metrics_file).group(1))
            df = self.spike_sorting_metrics(metrics_file)

            self.workbook.append(
                {
                    'section_id': section_parent,
                    'spike_sorting_metrics': df,
                    'blocks': []
                }
            )

        # match experiment (block) objects to section
        for matched_files in list(matched_block_files):
            # a regex to get the experiment identity
            ex_r = r'[A-Z]{2}_M\d+_Section_\d+_BLK\d+' 
            experiment_id = re.search(ex_r, matched_files[0]).group(0)
            
            section_child = int(re.search(r'Section_(\d{1,})', matched_files[0]).group(1))
            block = int(re.search(r'BLK(\d{1,})', matched_files[0]).group(1))

            experiment = load_experiment(*matched_files)
            self.workbook[section_child - 1]['blocks'].append({
                'block_id': block,
                'experiment': experiment
            })
            print(f"Sucessfully loaded {experiment_id}.")


# Class Errors
class _UnrecognizedNorm(Exception):
    """
    Exception raised for unrecognized user input
    for array normalization.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class _NoStimulusCondition(Exception):
    """
    Exception raised for no user input in the
    stim_condition argument in the 
    self.population_response method.
    """

    def __init__(self):
        self.message = """
        Please pass a stimulus condition or enter 'all' to generate 
        a joint condition population response matrix.
        """
        super().__init__(self.message)


class _UnrecognizedStimulusCondition(Exception):
    """
    Exception raised for invalid user input in the
    stim_condition argument in the 
    self.population_response method.
    """

    def __init__(self):
        self.message = """
        The stimulus condition does not exist within this experiment.
        Please enter a valid stimulus condition.
        """
        super().__init__(self.message)


class _UnrecognizedColumnsInput(Exception):
    """
    Exception raised for invalid user input in the
    columns argument in the 
    self.population_response method.
    """

    def __init__(self, arg):
        self.message = f"""
        Invalid input for 'columns'. Please select one of: {arg}.
        """
        super().__init__(self.message)
