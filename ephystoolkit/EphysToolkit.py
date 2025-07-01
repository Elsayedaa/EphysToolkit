# class imports
import re
import math
import json
import warnings
import os.path
import csv
import mat73

import numpy as np
import scipy.io
import pandas as pd
import rhd
from scipy import stats
from scipy import signal

from pathexplorer.PathExplorer import path_explorer


class ephys_toolkit:

    def __init__(self):
        
        self._SAMPLING_RATE = 20000
        self._modpath = os.path.dirname(__file__)

    def _bin_events(self, bin_size, events):
        self.frames = bin_size ** -1
        self.numerator = self._SAMPLING_RATE / self.frames

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
    
    def apply_temporal_smoothing(self, x, k, t_axis):
        return np.apply_along_axis(
                        lambda m: np.convolve(
                            m, k, 
                            mode='same'
                        ), axis=t_axis, arr=x
                    )
    
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
        
        # phi
        phi = np.radians(phi)

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

    def _gaussian_radius(self, dim:tuple, radius:int):
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

    def _linear_extrapolation(self, xd, yd, x):
        y2, y1 = yd[-1], yd[-2]
        x2, x1 = xd[-1], xd[-2]

        return y1+(((x-x1)/(x2-x1))*(y2-y1))

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
        self.stim_m73 = None
        self.spike_m73 = None
        self.spikefile = spikefile # path to the spike file
        self.stimfile = stimfile # path to the stim file
        
        try:
            self.spikes_mat = scipy.io.loadmat(spikefile)
            self.spikes = self.spikes_mat['Data'][0] # raw spikes dictionary
        except NotImplementedError:
            self.spike_m73 = True
            self.spikes_mat = mat73.loadmat(spikefile)
            self.spikes = self.spikes_mat['Data']
            
        try:
            self.stims_mat = scipy.io.loadmat(stimfile)
            self.stims = self.stims_mat['StimulusData'][0][0] # raw stims dictionary
        except NotImplementedError:
            self.stim_m73 = True
            self.stims_mat = mat73.loadmat(stimfile)
            self.stims = self.stims_mat['StimulusData']
            
        self.logfile = logfile # stimulus log data
        self.nodf = False
        self._init_stim_data()
        self._init_spike_data()
        self.set_time_unit()

        self.parameters_matched = False
        
        if self.nodf:
            self.stim_conditions = np.unique(
                self.stim_data['stim_condition_ids']
            )
        else:
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
        
        if self.stim_m73:
            stim_data = {
                'stim_start_indicies': self.stims['stimulusOnsets'],
                'stim_stop_indicies': self.stims['stimulusOffsets'],
                'stim_condition_ids': self.stims['conditionNumbers']
            }
        else:
            stims_starts = self.stims[0]
            stims_stops = self.stims[1]
            stims_condition_ids = self.stims[2]
            stim_data = {
                'stim_start_indicies': stims_starts[0],
                'stim_stop_indicies': stims_stops[0],
                'stim_condition_ids': stims_condition_ids[0]
            }
        
        try:
            self.stim_data = pd.DataFrame(stim_data)
        except ValueError as e:
            if str(e) == "All arrays must be of the same length":
                warn_text = f"""
                Stimulus start time, stop time, and condition 
                arrays are unequal in length. This typically
                only happens with natural image recordings.
                If you include a stimulus log file in the data
                folder, ephystoolkit will attempt to fix this.
                Otherwise, the .stim_data attribute will return
                a dictionary instead of a dataframe.
                """
                warnings.warn(warn_text, stacklevel = 4)
                self.stim_data = stim_data
                self.nodf = True

    # Generates the .spike_data attribute.
    def _init_spike_data(self):
        if self.spike_m73:
            ci = [int(i) for i in self.spikes['ChannelID']]
            si = self.spikes['SpikePoints']
            st = self.spikes['SpikeTimes']
            self.spike_data = [
                {
                    'channel_id': ci[i],
                    'spike_index': si[i],
                    'spike_time': st[i]
                }
                for i in range(len(st))
            ]
        else:
            
            self.spike_data = [
                {
                    'channel_id': unit[1][0][0],
                    'depth': self.depth_data.loc[self.depth_data.channel == unit[1][0][0]]['distance'][0],
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
            
            # Attempt to fix unequal stimulus arrays/missing values
            if self.nodf:
                stim_start_indicies = self.stim_data['stim_start_indicies']
                stim_stop_indicies = self.stim_data['stim_stop_indicies']
                stim_start_times = self.stim_data['stim_start_times']
                stim_stop_times = self.stim_data['stim_stop_times']
            else:      
                stim_start_indicies = self.stim_data['stim_start_indicies'].values
                stim_stop_indicies = self.stim_data['stim_stop_indicies'].values
                stim_start_times = self.stim_data['stim_start_times'].values
                stim_stop_times = self.stim_data['stim_stop_times'].values

            stim_data_fixed = {
                'stim_condition_ids': condition_ids,
                'stim_start_indicies': stim_start_indicies,
                'stim_stop_indicies': stim_stop_indicies,
                'stim_start_times': stim_start_times,
                'stim_stop_times': stim_stop_times
            }
            keys = list(stim_data_fixed.keys())

            # extrapolate the missing values
            for key in keys:
                xd = np.arange(len(stim_data_fixed[key]))+1
                yd = stim_data_fixed[key]
                x = np.arange(len(yd), len(condition_ids))+1
                y_pred = self._linear_extrapolation(xd, yd, x)
                if 'indicies' in key:
                    y_pred = np.round(y_pred)
                stim_data_fixed[key] = np.concatenate((yd, y_pred))
            
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
        if self.nodf:
            self.stim_data['stim_start_times'] = self._bin_events(bin_size,
                self.stim_data['stim_start_indicies'])

            self.stim_data['stim_stop_times'] = self._bin_events(bin_size,
                self.stim_data['stim_stop_indicies'])  
        else:
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

        for i, unit in enumerate(self.spike_data):
            try:
                unit.update({
                    'rel_spike_time': self._bin_events(bin_size,
                                                       unit['spike_index'])
                })
            except TypeError as e:
                if 'NoneType' in str(e):
                    print(f"No spikes recorded for unit {i}, skipping...")
                else:
                    raise TypeError

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
        
        # if parameters are given in a log file
        if os.path.splitext(params_file)[1] == '.log':
            df = self.stim_data
            with open(params_file, 'r') as f:
                lines = f.readlines()[1:-2]
            
            # read the lines to make a series of columns to insert
            insert = []
            for line in lines:
                items = re.split(r'\t+', line)[2:-1]
                dic = {
                    item.split(':')[0]:float(item.split(':')[1].strip(' ')) 
                    for item in items
                }
                insert.append(dic)
            
            # add the parameters to the stim_data dataframe
            insert = pd.DataFrame(insert)
            newcol = [list(df.columns)[0]] + list(insert.columns) + list(df.columns)[1:]
            self.stim_data = df.join(insert)[newcol]
            
            # make the parameter map
            stop_index = list(self.stim_data.columns).index('stim_start_indicies')
            df_new = self.stim_data[list(self.stim_data.columns)[:stop_index]]
            df_new.columns = ['condition']+list(df_new.columns)[1:]
            df_new = df_new.drop_duplicates().sort_values(by = 'condition')
            self.parameter_map = df_new.reset_index().drop('index', axis = 1)
            
            self.parameters_matched = True
            
        # if parameters are given in a matlab file        
        elif os.path.splitext(params_file)[1] == '.mat':
            
            # load the parameter file and extract the keys
            params = scipy.io.loadmat(params_file)
            param_keys = list(params.keys())

            # regex to get the parameter names + values
            name_identifier = r'Var\d{1,}Val'
            val_identifier = r'var\d{1,}value'

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
    
    def map_rf(self, rfunc, xpix, ypix, xvis, yvis, radius):
        self._stim_frame_map(xpix, ypix, xvis, yvis, radius)
        fmap = np.array([x for x in self.frame_map.values()])
        if fmap.T.shape[-1] == rfunc.shape[0]:
            return np.dot(fmap.T, rfunc)
        else:
            raise _InvalidRfuncShape((fmap.T.shape[-1], rfunc.shape[0]))
                    
    def _stim_frame_map(self, xpix, ypix, xvis, yvis, radius):
        frame_map = {}
        for i in range(len(self.parameter_map)):
            con = i+1
            dx = self.parameter_map.iloc[i]
            
            frame = self.static_grating(
                xpix, ypix,
                xvis, yvis,
                float(dx['Spatial Freq']), 
                float(dx['Orientation']), 
                float(dx['Phase'])*360,
                diameter = radius
                )

            frame_map[con] = frame

        self.frame_map = frame_map

    def _stim_frames(self):
        cond = self.stim_data.stim_condition_ids.values
        self.stim_frames = np.array([self.frame_map[i] for i in cond])

class lfp_tools(ephys_toolkit):

    #### initialize class with the necessary paths ###
    def __init__(self):
        ephys_toolkit.__init__(self)
        self.low = 1
        self.high = 120
        
    ### process the lfp data by running the necssary methods ###
    def process_lfp(self, intan_file, probe = None):
        """
        This method runs the lfp processing pipeline and
        generates two important attributes:
        
        - .lfp_heatmaps: contains a dictionary the heatmaps of the 
          lfp data for each channel column and each stimulus contrast.
          
        - .depth_data: contains the depth in micrometeres of each 
          channel normalized to the center of layer 4 as 0. 
          Negative values are ventral to layer 4 and positive values
          are dorsal to layer 4.
        
        """
        self.probe_fol = os.path.join(self._modpath, 'probes')
        valid_probes0 = os.listdir(self.probe_fol)
        valid_probes = [os.path.splitext(x)[0] for x in valid_probes0]
        if probe == None:
            inputtext = fr"""
            Please enter the probe used in this project. Available probes include: 
            {valid_probes}. 
            If the probe used in your recording is not included in the list, 
            navigate to the folder where the ephystoolkit lirbary is installed. 
            
            If you are using Anaconda on windows, this folder should be located at:
            C:\Users\(username)\conda\envs\(env_name)\lib\site-packages\ephystoolkit
            
            If you are using Anaconda on Linux, this folder should be located at:
            ~/anaconda3/envs/(env_name)/lib/(Python_version)/site-packages/ephystoolkit'
            
            Once you are in the module folder, navigate the to folder called 'probes'
            and copy a csv file mapping your channel ids to their distance from the tip
            of the probe. Each line index should correspond to the channel id. Indexes
            for this purpose start at 1. For reference, check out any of the included
            csv files."""
            
            probe = input(inputtext)
        while probe not in valid_probes:
            inputtext = fr"""
            {probe} is not included in the list of available probes. The list of available 
            probes includes: 
            {valid_probes}
            If the probe used in your recording is not included in the list, 
            navigate to the folder in which the ephys_toolkit module is installed. 

            If you are using Anaconda on windows, this folder should be located at:
            C:\Users\(username)\conda\envs\(env_name)\lib\site-packages\ephystoolkit

            If you are using Anaconda on Linux, this folder should be located at:
            ~/anaconda3/envs/(env_name)/lib/(Python_version)/site-packages/ephystoolkit

            Once you are in the module folder, navigate the to folder called 'probes'
            and copy a csv file mapping your channel ids to their distance from the tip
            of the probe. Each line index should correspond to the channel id. Indexes
            for this purpose start at 1. For reference, check out any of the included
            csv files.
            """
            probe = input(inputtext)
        else:
            pass
        
        self.probe = probe
        self.intan_file = intan_file
        self.channel_file = os.path.join(self.probe_fol, f"{probe}.csv")
        
        self._load_lfp_data()
        
        print("Sorting channel depth...")
        self._sort_depth()
        
        print("Retrieving stimulus onset times...")
        self._get_onsets()
        
        print("Generating lfp heatmap...")
        self._get_lfp_heatmap()
        
        print("Normalizing channel depth to layer 4 depth...")
        self._find_l4_distances()
        self._normalize_depth()
        print("Done!")
    
    ### bandpass filter for the raw voltage data ###
    def _bandpass_filter(self, signal):
        
        #SAMPLING_RATE = 20000.0
        nyqs = 0.5 * self._SAMPLING_RATE
        low = self.low/nyqs
        high = self.high/nyqs

        order = 2

        b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog = False)
        y = scipy.signal.filtfilt(b, a, signal, axis=0)

        return y

    ### load the necessary data ###
    def _load_lfp_data(self):
        
        # load the rhd file
        self.intan_data = data = rhd.read_data(self.intan_file)
        self.amplifier_data = self.intan_data['amplifier_data']
        self.board_data = self.intan_data['board_dig_in_data']

        # load the channel file
        self.channel_data = []
        with open(self.channel_file, newline='') as channels:
            for row in csv.reader(channels):
                self.channel_data.append(np.array(row).astype(float))
        self.channel_data = np.array(self.channel_data)

        # bandpass filter each channel
        print("Bandpass filtering the data...")
        self.amplifier_data_filt = np.array([
            self._bandpass_filter(channel) for
            channel in self.amplifier_data
        ]
        )
        
    ### sort amplifier data by channel depth ###
    def _sort_depth(self):
        ## 0 = deepest
        
        # adjust single channels offset on x by a negligable amount
        if self.probe == '64D':
            self.channel_data[np.where(self.channel_data == 16)] = 20
            self.channel_data[np.where(self.channel_data == -16)] = -20
        if self.probe == '64H':
            self.channel_data[np.where(self.channel_data == 20)] = 22.5
            self.channel_data[np.where(self.channel_data == -20)] = -22.5
            self.channel_data[np.where(self.channel_data == 180.1)] = 177.6
            self.channel_data[np.where(self.channel_data == 220.1)] = 222.6
        
        # index ordered by distance from tip
        self.y_index = np.argsort(self.channel_data[:,1])[::-1]
        
        # list of channel depths ordered by distance from tip
        self.c_depth = self.channel_data[self.y_index][:,1]

        # unique x coordinates for the channels
        xarr = np.unique(self.channel_data[:,0])
        
        # get channel depths along specific columns
        self.channels_bycol = {}
        for x_d in xarr: 
            col = np.where(self.channel_data[self.y_index][:,0] == x_d)[0]
            self.channels_bycol[x_d] = {
                'y_index': self.y_index[col],
                'c_depth': self.channel_data[self.y_index[col]][:,1]
            }
        
        # get amplifier data along specific columns
        self.ampdata_bycol = {}    
        for col, subdict in self.channels_bycol.items():
            self.ampdata_bycol[col] = self.amplifier_data_filt[
                subdict['y_index']
            ]

    
    ### Get the onset times of each stimuls contrast ###
    def _get_onsets(self):
        
        # get indices of when contrast 0 is on
        stim0 = self.board_data[1]
        stim0_on_index = (np.where(stim0 == True)[0])

        # get indices of when contrast 1 is on
        stim1 = self.board_data[2]
        stim1_on_index = (np.where(stim1 == True)[0])

        # get the onset indices of contrast 0
        first0_on = stim0_on_index[0]
        self.c0_onsets = [first0_on]
        for i0, val in enumerate(stim0_on_index[1:]):
            i1 = i0+1
            if stim0_on_index[i0]+1 != (stim0_on_index[i1]):
                self.c0_onsets.append(val)

        # get the onset indices of contrast 1
        first1_on = stim1_on_index[0]
        self.c1_onsets = [first1_on]
        for i0, val in enumerate(stim1_on_index[1:]):
            i1 = i0+1
            if stim1_on_index[i0]+1 != (stim1_on_index[i1]):
                self.c1_onsets.append(val)
                
    ### get the heatmap of lfp data averaged across stimulus repeats ###
    def _get_lfp_heatmap(self):

        self.lfp_heatmaps = {}
        for col, subdict in self.ampdata_bycol.items():
            self.lfp_heatmaps[col] = {
                'contrast0': np.array([
                    subdict[:,i:i+5000] for i in self.c0_onsets
                ]).mean(0),
                'contrast1': np.array([
                    subdict[:,i:i+5000] for i in self.c1_onsets
                ]).mean(0),

            }
        
    ### assign a depth to each channel relative to layer 4 ###
    def _find_l4_distances(self):
        
        self.l4_distances = {}
        self.l4_distance_list = []
        for col, chandata in self.channels_bycol.items():
            lfp_heatmap = self.lfp_heatmaps[col]
            self.l4_distances[col] = {}
            for c, contrast in lfp_heatmap.items():
            
                # find the timepoint of the deepest sink
                pe_delay = 1000
                tmin = np.where(contrast[:,pe_delay:] == contrast[:,pe_delay:].min())[1][0]
                tmin_curve = contrast[:,tmin+pe_delay]

                ## Find the center of layer 4 via cubic spline interpolation

                # flip the order because the interpolation function
                # strictly requires an x axis with ascending order
                x = chandata['c_depth'][::-1]
                y = tmin_curve

                # initialize the interpolation function
                cs = scipy.interpolate.CubicSpline(x, y)

                # interpolate the data and flip it back to the correct order
                xnew = np.linspace(0, x[-1], 1575)[::-1]
                ynew = cs(xnew)[::-1]

                # get the layer 4 distance from tip
                l4_distance = xnew[np.where(ynew == ynew.min())[0][0]].round(2)
                self.l4_distances[col][c] = l4_distance
                self.l4_distance_list.append(l4_distance)
        self.l4_distance_list = np.array(self.l4_distance_list)
    
    def _normalize_depth(self):
        
        # find the average distance across channel
        # columns and stimulus contrasts
        avg_distance = self.l4_distance_list.mean()

        # normalize channel distances by distance from layer 4
        # negative distances = below layer 4
        # positive distances = above layer 4
        l4_normalization = self.c_depth - avg_distance

        # compile the depth data into a dataframe
        depth_data = {
            'channel': self.y_index,
            'distance': l4_normalization
        }
        self.depth_data = pd.DataFrame(depth_data)        

class load_project(lfp_tools):
    """
    Initialize the load_project class with a full path to the
    directory containing the project files. If the LFP data rhd
    file is included in the project directory, this class will 
    automatically process the LFP data for the project. 
    
    The .workbook attribute contains a list of dictionaries
    with the following structure:
    

      
          {
          
              'section_id': int - Identification number of the recording 
               section,
              
              'spike_sorting_metrics': Pandas dataframe - Contains spike
               sorting metrics data,
              
              'lfp_heatmaps': dictionary - Contains the lfp heatmap
               at each channel column and stimulus contrast,
              
              'depth_data': Pandas dataframe - Contains the distance of each
               channel of the probe relative to layer 4,
              
              'blocks': [
              
                  {
                  
                  'block_id': int - Identification number of the recording block, 
                  
                  'experiment', experiment object - Contains the experiment data for
                   the given block
                  
                  },
                  
                  ]
                  
          },
    
    Args:
    
    - project_path: Path to the directory containing the project files.
    """

    def __init__(self, project_path, probe = None, use_lfp_file = 0):
        lfp_tools.__init__(self)
        self.lfp_index = use_lfp_file
        self.probe = probe
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
        metrics_files.sort(key=lambda x: int(re.search(r'Section_(\d{1,})', x).group(1)))
        
        # find checkerboard rhd files
        rhd_files = explorer.findext(self.ppath, '.rhd')
        rhd_files.sort(key=lambda x: int(re.search(r'Section_(\d{1,})', x).group(1)))
        
        # group together multiple checkerboard recordings
        # from the same section
        i = 0
        for rhd_file in rhd_files[1:]:
            if type(rhd_files[i])!=list:
                r = r"[A-Z][A-Z]_M\d{1,}_Section_(\d)"
                id_sec_post = re.search(r, rhd_file).group(0)
                id_sec_pre = re.search(r, rhd_files[i]).group(0)
                if id_sec_post == id_sec_pre:
                    rhd_files[i] = [rhd_files[i], rhd_file]
                    post_index = rhd_files.index(rhd_file)
                    del rhd_files[post_index]

            elif (type(rhd_files[i])==list):
                id_sec_post = re.search(r, rhd_file).group(0)
                id_sec_pre = re.search(r, rhd_files[i][0]).group(0)
                if id_sec_post == id_sec_pre:
                    rhd_files[i].append(rhd_file)
                    post_index = rhd_files.index(rhd_file)
                    del rhd_files[post_index]
            else:
                i+=1

        ################################################################################

        # compile the workbook
        self.workbook = []
        
        # compile metrics
        for metrics_file in metrics_files:
            section_parent = int(re.search(r'Section_(\d{1,})', metrics_file).group(1))
            df = self.spike_sorting_metrics(metrics_file)
            self.workbook.append(
                {
                    'section_id': section_parent,
                    'spike_sorting_metrics': df,
                    'lfp_heatmaps': None,
                    'depth_data': None,
                    'blocks': []
                }
            )
            
        # compile rhd
        if rhd_files == []:
            warn_text = """
            No LFP data files found in this project folder. LFP data will be
            unavailable for this project. Please include the LFP data rhd file 
            in the project folder if you want to access the LFP data.
            """
            warnings.warn(warn_text, stacklevel = 4)
        else:
            for rhd_file in rhd_files:
                if (type(rhd_file) == list) & (self.lfp_index == 0):
                    warn_text = """
                    WARNING: More than one LFP data file found in this section.
                    Defaulting to the first LFP file found in the section. If 
                    You wish to change which file to be used, pass an integer 
                    value to the use_lfp_file parameter of the load_project class
                    corresponding to the file you wish to use. 
                    (ie: load_project(use_lfp_file = 2) lets you use the 2nd LFP 
                    data file.).
                    """
                    warnings.warn(warn_text, stacklevel = 4)
                    
                    rhd_file = rhd_file[self.lfp_index]
                elif (type(rhd_file) == list) & (self.lfp_index != 0):
                    use_lfp_file = use_lfp_file-1
                    rhd_file = rhd_file[self.lfp_index]
                else:
                    pass
                
                self.process_lfp(rhd_file, self.probe)
                section_parent = int(re.search(r'Section_(\d{1,})', rhd_file).group(1))
                self.workbook[section_parent-1]['lfp_heatmaps'] = self.lfp_heatmaps
                self.workbook[section_parent-1]['depth_data'] = self.depth_data

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
        
class _InvalidRfuncShape(Exception):
    """
    Exception raised for invalid shape of the response
    function given for receptive field mapping.
    """

    def __init__(self, arg):
        self.message = f"""
        Shape mistmatch: Frame map dim - 1 ({arg[0]}) != Response function dim 0 ({arg[1]}).
        """
        super().__init__(self.message)