import numpy as np
from scipy.interpolate import interp1d
import polars as pl

def slice_data(df):
    
    # trunkate dfs
    _temp_df = df
    _start_inx = None
    _end_inx = None
    _trailing_trigger = False
    for i in range(len(_temp_df['sync'])):
        if (_temp_df['sync'][i] == 1) and (not _trailing_trigger):
            _start_inx = i
            _trailing_trigger = True
        
        if (_temp_df['sync'][i] == 0) and (_trailing_trigger):
            _end_inx = i
            break
    _temp_df = _temp_df[_start_inx:_end_inx]   
    return _temp_df

def interpolate(ref_df, target_df):
    # df is the df to be interpolated
    # ref_df is the reference df
    # df and ref_df should have the same columns

    _reference_time = ref_df['timestamp']

    columns = target_df.columns[1:-1]
    interpolated_cols = {}
    for _col in columns:
        interp_func = interp1d(target_df['timestamp'], target_df[_col], kind='linear', axis=0, fill_value='extrapolate')
        interpolated_cols[_col] = interp_func(_reference_time)
        
    df = pl.from_dict(interpolated_cols).insert_column(0, _reference_time)
    return df

def interpolats_dfs(dfs, ref_no = 0):
    # generally first df is the reference df
    _keys = list(dfs.keys())
    _reference_df = dfs[_keys[ref_no]]
    _temp_dfs = dfs.copy()
    _temp_dfs.pop(_keys[ref_no])
    _selected_keys = _temp_dfs.keys()    
    _new_dfs = {}
    for _key in _selected_keys:
        _new_dfs[_key] = interpolate(_reference_df, dfs[_key]) 
        
    _new_dfs[_keys[ref_no]] = _reference_df
    
    return _new_dfs

def preprocess(dfs):
    for _name in dfs.keys():
        dfs[_name] = slice_data(dfs[_name])
        
    dfs = interpolats_dfs(dfs)
        
    return dfs

def apply_median_filter(dfs):
    return dfs