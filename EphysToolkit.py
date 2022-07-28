# class imports
import sys
import platform
import re
import numpy as np
import json
import scipy.io
import pandas as pd
from scipy import stats

if platform.system() == 'Linux':
    sys.path.append("/media/sf_DebianShared/custom_modules")
if platform.system() == 'Windows':
    sys.path.append(r"C:\Users\AE\Documents\DebianShared\custom_modules")
from PathExplorer import path_explorer


class ephys_toolkit:
    
    def __init__(self):
        self.SAMPLING_RATE = 20000
    
    """
    Give a bin size relative to 1 second.
    IE: If you want a 1 ms bin size, enter 0.001;
    if you want a 10 ms bin size, enter 0.01 etc.
    """
    def bin_events(self, bin_size, events):
        self.frames = bin_size**-1
        self.numerator = self.SAMPLING_RATE/self.frames
        
        return events/self.numerator
    
    def minmax_norm(self, array):
        return (array-array.min())/(array.max()-array.min())
    
    def zscore_norm(self, array):
        return stats.zscore(array)
    
    def average_response(self, array, stim_reps):
        return (array/stim_reps)*self.frames
    
    def make_grating_matrix(
            self,
            sf, 
            ori, 
            ph, 
            dim = tuple,
            radius = int,
            edge = 'discrete'):

        edge_filt = {
            'discrete': self.discrete_radius(dim, radius),
            'gaussian': self.gaussian_radius(dim, radius)
        }
        filt = edge_filt[edge]

        step = 20/dim[0]

        x = np.arange(-10,10,step)
        x = np.array([x for i in range(dim[0])])
        y = x.T

        fx = sf*np.cos(np.radians(ori))
        fy = sf*np.sin(np.radians(ori))

        m = np.cos(2*np.pi*fx*x + 2*np.pi*fy*y + np.radians(ph)*2*np.pi)

        return m*filt

    #make a drifting grating from a series of stationary gratings
    def make_drifting_grating_matrix(
            self, 
            sf, 
            ori, 
            tf, 
            dt, 
            t, 
            dim = tuple,
            radius = int,
            edge = 'discrete'):

        tensor = []
        params = []
        deg_step = dt*360*tf

        d = np.arange(dt,t,dt)

        phase = 0
        for x in d:
            m = self.make_grating_matrix(
                sf, 
                ori, 
                phase, 
                dim,
                radius,
                edge)
            tensor.append(m)
            params.append([sf, tf, ori, phase])
            phase+=deg_step 

        return tensor, params

    def discrete_radius(self, dim = tuple, radius  = int):
        x, y = np.meshgrid(
            np.linspace(-1,1, dim[0]), 
            np.linspace(-1,1, dim[1])
        )

        m = []
        for i0 in range(len(x)):
            row = []
            for i1 in range(len(y)):
                if ((x[i0,i1]**2 + y[i0,i1]**2) 
                    < ((radius/360)**2)):
                        row.append(1)
                else:
                    row.append(0)
            m.append(row)

        return np.array(m)
    
    def gaussian_radius(self, dim = tuple, radius = int):
        x, y = np.meshgrid(
            np.linspace(-1,1, dim[0]), 
            np.linspace(-1,1, dim[1])
        )
        d = np.sqrt(x*x+y*y)
        sigma, mu = radius/360, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

        return g
    
    def get_spike_sorting_metrics(self, file_directory):
        
        with open(file_directory, 'r') as sorting_file:
            sorting_info = json.load(sorting_file)
        
        spike_sorting_data = [
            #[cluster['label'] for cluster in sorting_info['clusters']],
            [cluster['metrics']['isolation'] for cluster in sorting_info['clusters']],
            [cluster['metrics']['noise_overlap'] for cluster in sorting_info['clusters']]
        ]
        
        ss_df = pd.DataFrame(np.array(spike_sorting_data).T)
        ss_df.columns = [
            # 'cluster', 
            'isolation', 'noise_overlap'
        ]
        
        return ss_df
          
    #functions to make concatenated across trial and non concatenated across trial rasters
    def concatenated_raster(self, stims, spikes, thresh = tuple):
        if thresh == tuple:
            r = np.array([spikes-st for st in stims])
            raster = np.concatenate(r)
        else:
            r = np.array([spikes-st for st in stims])
            ti = np.where(np.logical_and(r <= thresh[1], r >= thresh[0]))
            raster = r[ti]
        return raster

    def unconcatenated_raster(self, stims, spikes, thresh = tuple):
        if thresh == tuple:
            rasters = np.array([spikes-st for st in stims])
        else:
            rasters = []
            for i, st in enumerate(stims): #enumerate to make an initial array then vstack
                unthreshed = spikes-st
                i = np.where(np.logical_and(unthreshed <= thresh[1], unthreshed >= thresh[0]))
                rasters.append(list(unthreshed[i]))
        return rasters

    def make_raster(self, stims, spikes, thresh = tuple, concatenate = True):
        if concatenate == True:
            return self.concatenated_raster(stims, spikes, thresh)
        else:
            return self.unconcatenated_raster(stims, spikes, thresh)

class load_experiment(ephys_toolkit):
    
    def __init__(self, spikefile, stimfile):
        ephys_toolkit.__init__(self)
        
        spikes_mat = scipy.io.loadmat(spikefile)
        stims_mat = scipy.io.loadmat(stimfile)
        
        self.spikes = spikes_mat['Data'][0]
        self.stims = stims_mat['StimulusData'][0][0]
        self.init_stim_data()
        self.parameters_matched = False
        self.init_spike_data()
        self.get_event_times()
        
        self.stim_conditions = self.stim_data[
            'stim_condition_ids'
        ].unique()
    
    def init_stim_data(self):
        stims_starts = self.stims[0]
        stims_stops = self.stims[1]
        stims_condition_ids = self.stims[2]
        
        stim_data = {
            'stim_start_indicies':stims_starts[0],
            'stim_stop_indicies':stims_stops[0],
            'stim_condition_ids':stims_condition_ids[0]
        }
        
        self.stim_data = pd.DataFrame(stim_data)
        
    def init_spike_data(self):
        
        self.spike_data = [
            {
                # 'cluster_id':unit[0][0][0],
                'channel_id':unit[1][0][0],
                'spike_index':unit[2][0],
                'spike_time':unit[3][0]
            }
            for unit in
            self.spikes
        ]
    
    def get_event_times(self, bin_size = 0.001):
        self.stim_data['stim_start_times'] = self.bin_events(bin_size, 
            self.stim_data['stim_start_indicies'].values)
        
        self.stim_data['stim_stop_times'] = self.bin_events(bin_size, 
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
                'rel_spike_time': self.bin_events(bin_size, 
                    unit['spike_index'])
            }) 
    
    def get_condition_times(self, group):
        
        condition_starts = self.stim_data.groupby(
            self.stim_data.stim_condition_ids
        ).get_group(group)['stim_start_times'].values
        
        condition_stops = self.stim_data.groupby(
            self.stim_data.stim_condition_ids
        ).get_group(group)['stim_stop_times'].values
        
        return {
            'start': condition_starts,
            'stop': condition_stops
        }

    def match_condition_parameters(self, params_file):
        # regex to get the parameter names + values
        name_identifier = r'Var\d{1,}Val'
        val_identifier = r'var\d{1,}value'    

        # load the parameter file and extract the keys
        params = scipy.io.loadmat(params_file)
        param_keys = list(params.keys())

        # use regex to get the param names and values
        name_keys = [key for key in param_keys if re.search(name_identifier, key)]
        val_keys = [key for key in param_keys if re.search(val_identifier, key)]

        # get the condition numbers
        condition_range = len(params[val_keys[0]][0])
        condition_ids = [i+1 for i in range(condition_range)]  

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
        for cond  in df.stim_condition_ids.values:
            arr = self.parameter_map.loc[self.parameter_map.condition == cond].values[0]
            insert.append(arr)
        insert = pd.DataFrame(insert, columns = self.parameter_map.columns.values)

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
   
    def single_population_response_matrix(
            self,
            include_units,
            stim_condition_start_times,
            thresh = None,
            norm = None):
        
        if thresh == None:
            return "Please provide a response boundary threshhold as a tuple (ex: (-50,500))"
        else:
            thresh_min = thresh[0]-0.5
            thresh_max = thresh[1]+0.5

        stb = {}
        self.non_responsive = []
        
        for i in include_units:

            unit = self.spike_data[i]
            cluster_id = i
            x = self.make_raster(stim_condition_start_times, unit['rel_spike_time'], thresh = thresh)
            h, bins = np.histogram(x, bins = np.arange(
                thresh_min,
                thresh_max,
                1
            ))

            if h.min() == 0 and h.max() == 0:
                self.non_responsive.append(i)
                h = np.zeros(len(h))
                
            else:
                norm_method = {
                    'minmax': self.minmax_norm(h), 
                    'zscore': self.zscore_norm(h),
                    'average' :self.average_response(h, len(stim_condition_start_times))
                }

                if norm == None:
                    feedback = "Population response matrix generated with NO NORMALIZATION."
                    
                elif (norm != None
                          and norm in list(norm_method.keys())):
                    h = norm_method[norm]
                    feedback = f"Population response matrix generated with {norm.upper()} NORMALIZATION."
                else:
                    raise UnrecognizedNorm(
                        f"Normalization method is not recognized, please choose from the following: {list(norm_method.keys())}"
                    )

            stb[cluster_id] = h

        stb = pd.DataFrame(stb)
        
        return stb
    
    def joint_population_response_matrix(self, include_units, thresh, norm):
        # get the population response matrix for the first stim condition
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        
        first_stim_start_times = self.get_condition_times(
            self.stim_conditions[0])['start']

        con_df = self.single_population_response_matrix(
            include_units, 
            first_stim_start_times, 
            thresh = (thresh_min, thresh_max), 
            norm = norm)

        # define the remaining stim conditions
        remaining_stim_conditions = self.stim_conditions[1:]

        # concatenate the population response matricies for the
        # remaining stim conditions
        for condition in remaining_stim_conditions:
            stim_start_times = self.get_condition_times(
                condition)['start']

            df = self.single_population_response_matrix(
                include_units, 
                stim_start_times, 
                thresh = (thresh_min, thresh_max), 
                norm = norm)

            con_df = pd.concat(
                [con_df, df],
                axis = 0, 
                ignore_index = True)
            
        if self.parameters_matched == False:
            # make a conditions column
            bins_per_condition = len(np.arange(thresh_min, thresh_max, 1))
            con_df_columns = list(con_df.columns)
            conditions_column = []
            for condition in self.stim_conditions:
                conditions_column += [condition]*bins_per_condition

            # arrange dataframe column order
            con_df['stimulus_condition'] = np.array(conditions_column)
            new_columns = ['stimulus_condition'] + con_df_columns
            con_df = con_df[new_columns]
            
        
        else: # repeat what's above while adding parameter columns    

            bins_per_condition = len(np.arange(thresh_min, thresh_max, 1))
            con_df_columns = list(con_df.columns)
            conditions_column = []
            param_columns = []
            
            for condition in self.stim_conditions:
                params = list(
                    self.parameter_map.loc[self.parameter_map.condition == condition].values[0][1:]
                )
                conditions_column += [condition]*bins_per_condition
                param_columns += [params]*bins_per_condition
            param_columns = pd.DataFrame(
                param_columns, columns = self.parameter_map.columns.values[1:]
            )
            
            con_df['stimulus_condition'] = np.array(conditions_column)
            con_df = con_df.join(param_columns)
            new_columns = ['stimulus_condition'] + list(param_columns.columns.values) + con_df_columns
            con_df = con_df[new_columns]
        
        return con_df
    
    # needs to be cleaned up and optimized...but it works
    # caching matricies that have already been generated 
    # might be a good idea
    def get_population_response_matrix(
            self,
            include_units, 
            stim_condition = None,
            columns = 'cluster_id',
            thresh = None, 
            norm = None):
        
        
        if stim_condition == None:
            raise NoStimulusCondition()
            
        elif stim_condition == 'all':
            single = False
            stb = self.joint_population_response_matrix(
                include_units,
                thresh,
                norm)
        
        elif stim_condition in self.stim_conditions:
            single = True
                
            stim_start_times = self.get_condition_times(stim_condition)['start']
            
            stb = self.single_population_response_matrix(
                include_units,
                stim_start_times,
                thresh,
                norm)
        else:
            raise UnrecognizedStimulusCondition()
        
        
        try: # remove the if statement when implementing caching
            if columns != 'stimulus_condition':
                stb_cid_col = None
            else:
                if self.parameters_matched == True:
                    i = len(self.parameter_map.columns)
                    c = list(stb.columns.values[i:])
                    stb1 = stb[['stimulus_condition']+c]
                else:
                    stb1 = stb

                stb_cid_col = pd.melt(

                    stb1, 
                    id_vars=['stimulus_condition'], 
                    value_vars= stb1.columns.values[1:]).pivot_table(

                        columns = 'stimulus_condition', 
                        index = 'variable', 
                        values = 'value', 
                        aggfunc = list)

                stb_cid_col_reset = stb_cid_col.reset_index()
                condition_list = list(stb_cid_col_reset.columns.values[1:])
                stb_cid_col_reset.columns = ['cluster_id']+condition_list
                stb_cid_col = stb_cid_col_reset.explode(condition_list).astype(float)

        except KeyError:
            if (single
                    and columns != 'cluster_id'):
                print("Not enough stimulus conditions")
                
            else:
                stb_cid_col = None
                    
        columns_dic = {
            'stimulus_condition': stb_cid_col,
            'cluster_id': stb}
        
        if columns in list(columns_dic.keys()):
            return columns_dic[columns]
                               
        else:
            raise UnrecognizedColumnsInput(list( columns_dic.keys())) 
    
    #Map receptive field using spike triggered averaging.
    def spike_triggered_rf(
            self, cluster,
            enlarge = 1,
            psize = 30.0,
            psf = 0.02,
            pph = 0.0,
            pori = 0.0,

    ):
        # Get the index range for stim appearnce
        stim_df = self.stim_data
        stim_app_range = list(zip(
            stim_df['stim_start_indicies'].values[:-1], 
            stim_df['stim_start_indicies'].values[1:]))
        stim_ranges = [
            np.arange(r[0], r[1], 1) for r in stim_app_range
        ]

        spike_ind = self.spike_data[cluster]['spike_index']
        final_start = stim_df['stim_start_indicies'].values[-1]
        final_range = np.arange(final_start, spike_ind[-1], 1)
        stim_ranges.append(final_range)

        stim_df['index_range'] = stim_ranges
        df = stim_df.explode('index_range')

        # Find the stimulus parameters prior to each spike.
        # By default, paremeters in stim_data will be used.
        # User declared parameters can also be used and can
        # either be passed as float values or 1d array-like 
        # structures. If parameters are not found in stim_data
        # or not passed by the user, default values for the
        # parameters will be used.

        first_stim = int(df.iloc[0]['stim_start_indicies'])
        t = []

        for i0, i in enumerate(spike_ind):
            prior_i = i-1
            if prior_i - first_stim < 0:
                pass
            else:
                s = df.iloc[prior_i - first_stim]

                # Size ######################## 
                if 'Size' in df.columns.values:
                    size = s['Size'] * enlarge
                elif type(psize) == float:
                    size = psize * enlarge
                else:
                    size = psize[i0] * enlarge

                # Spatial frequency ############
                if 'Spatial Freq' in df.columns.values:
                    sf = s['Spatial Freq']
                elif type(psf) == float:
                    sf = psf
                else:
                    sf = psf[i0]

                # Phase ########################
                if 'Phase' in df.columns.values:
                    ph = s['Phase']*360
                elif type(pph) == float:
                    ph = pph*360
                else:
                    pf = pph[i0]*360

                # Orientation ##################
                if 'Orientation' in df.columns.values:
                    ori = s['Orientation']
                elif type(pori) == float:
                    ori = pori
                else:
                    ori = pori[i0]

                # Make the pixel intensity matrix
                m = ephys_toolkit.make_grating_matrix(
                        sf, 
                        ori, 
                        ph, 
                        dim = (50,50),
                        radius = size,
                        edge = 'discrete')

                t.append(m)

        sta = np.mean(np.array(t), axis = 0)
        return sta  

class load_project(ephys_toolkit):
    
    # initialize the load_project class with a full path to the
    # directory containing the project files
    def __init__(self, project_path):
        ephys_toolkit.__init__(self)
        self.ppath = project_path
        self.gen_project_workbook()
    
    # generate the workbook of project data
    def gen_project_workbook(self):
        explorer = path_explorer()
        
        # find and sort spike files
        spike_files = explorer.findext(self.ppath,  '.mat', r = 'firings')
        spike_files.sort(key = lambda x: int(re.search(r'BLK(\d{1,})', x).group(1)))
        
        # find and sort stim files
        stim_files = explorer.findext(self.ppath,  '.mat', r = 'stimulusData')
        stim_files.sort(key = lambda x: int(re.search(r'BLK(\d{1,})', x).group(1)))
        
        # match spike and stim files
        matched_spike_stim = zip(spike_files, stim_files)
        
        #find metrics files
        metrics_files = explorer.findext(self.ppath,  '.json', r = 'metrics_isolation')
        
        ################################################################################
        
        # compile the workbook
        self.workbook = []
        
        # compile spike sorting metrics first
        for metrics_file in metrics_files:
            section_parent = int(re.search(r'Section_(\d{1,})', metrics_file).group(1))
            df = self.get_spike_sorting_metrics(metrics_file)

            self.workbook.append(
                {
                    'section_id': section_parent, 
                    'spike_sorting_metrics': df, 
                    'blocks': []
                }
            )
        
        # match experiment (block) objects to section
        for spike_stim in matched_spike_stim:
            
            section_child = int(re.search(r'Section_(\d{1,})', spike_stim[0]).group(1))
            block = int(re.search(r'BLK(\d{1,})', spike_stim[0]).group(1))

            experiment = load_experiment(*spike_stim)
            self.workbook[section_child-1]['blocks'].append({
                'block_id': block,
                'experiment': experiment
            })
    
#Class Errors
class UnrecognizedNorm(Exception):
    """
    Exception raised for unrecognized user input
    for array normalization.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NoStimulusCondition(Exception):
    """
    Exception raised for no user input in the
    stim_condition argument in the 
    self.get_population_response_matrix method.
    """
    def __init__(self):
        self.message = """
        Please pass a stimulus condition or enter 'all' to generate 
        a joint condition population response matrix.
        """
        super().__init__(self.message)

class UnrecognizedStimulusCondition(Exception):
    """
    Exception raised for invalid user input in the
    stim_condition argument in the 
    self.get_population_response_matrix method.
    """
    def __init__(self):
        self.message = """
        The stimulus condition does not exist within this experiment.
        Please enter a valid stimulus condition.
        """
        super().__init__(self.message)
                               
class UnrecognizedColumnsInput(Exception):
    """
    Exception raised for invalid user input in the
    columns argument in the 
    self.get_population_response_matrix method.
    """
    def __init__(self, arg):
        self.message = f"""
        Invalid input for 'columns'. Please select one of: {arg}.
        """
        super().__init__(self.message)                             
                              
