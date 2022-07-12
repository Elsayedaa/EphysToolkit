# class imports
import numpy as np
import scipy.io
import pandas as pd
import json
from scipy import stats

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
                'cluster_id':unit[0][0][0],
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
    
    def get_spike_sorting_metrics(self, file_directory):
        
        with open(file_directory, 'r') as sorting_file:
            sorting_info = json.load(sorting_file)
        
        spike_sorting_data = [
            [cluster['label'] for cluster in sorting_info['clusters']],
            [cluster['metrics']['isolation'] for cluster in sorting_info['clusters']],
            [cluster['metrics']['noise_overlap'] for cluster in sorting_info['clusters']]
        ]
        
        ss_df = pd.DataFrame(np.array(spike_sorting_data).T)
        ss_df.columns = ['cluster', 'isolation', 'noise_overlap']
        
        return ss_df

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
            cluster_id = float(unit['cluster_id'])
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
        
        return con_df
    
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
        
        try:
#             #Alternate solution - not formatting the index correctly
#             stb_cid_col = stb.groupby(

#                 stb.stimulus_condition).agg(list).T

#             stb_cid_col = stb_cid_col.explode(

#                 list(stb_cid_col.columns.values)
#             )

            stb_cid_col = pd.melt(

                stb, 
                id_vars=['stimulus_condition'], 
                value_vars= stb.columns.values[1:]).pivot_table(

                    columns = 'stimulus_condition', 
                    index = 'variable', 
                    values = 'value', 
                    aggfunc = list)

            stb_cid_col_reset = stb_cid_col.reset_index()
            condition_list = list(stb_cid_col_reset.columns.values[1:])
            stb_cid_col_reset.columns = ['cluster_id']+condition_list
            stb_cid_col = stb_cid_col_reset.explode(condition_list).astype(float)
            
#         except AttributeError:
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
                              
