'''
Interface for COIN model simulation.
'''

COIN_path = '/homes/ar2342/frogs_project/COIN_Python'
import sys
sys.path.append(COIN_path)

import numpy as np
from coin import COIN
from models import ModelForRunner
from types import SimpleNamespace
import pandas as pd


class COINWrapper(ModelForRunner):
    def __init__(self, **coin_kwargs):
        """
        Initialize the COINWrapper.

        Parameters:
            coin_kwargs: Optional keyword arguments to pass to COIN.
        """
        #removing 'info' from kwargs
        if 'info' in coin_kwargs:
            del coin_kwargs['info']
        self.coin_kwargs = coin_kwargs
        self.coin_model = COIN(**coin_kwargs)

    def reset_state(self):
        """
        Reset the state of the COIN model by instantiating a brand new COIN object.
        """
        self.coin_model = COIN(**self.coin_kwargs)

    def __call__(self, y):
        """
        Call the COIN model with input y.

        This method overrides the coin_model's perturbations with y and runs the simulation.
        It then maps the simulation output to a records object with:
         - records.u and records.u_lp set to the motor_output
         - records.state_feedback set to state_feedback

        Parameters:
            y: A NumPy array (or convertible) that overrides the COIN model's perturbations.

        Returns:
            A SimpleNamespace object with an attribute `records` containing the mapped simulation data.
        """
        # Override COIN model's perturbations with the provided input y
        # Ensure y is a NumPy array
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.coin_model.perturbations = y

        # Run the COIN simulation
        simulation_output = self.coin_model.simulate_coin()
        
        # Extract the first run from the simulation output
        # run = simulation_output["runs"][0]
        # motor_output = run["motor_output"]
        # state_feedback = run["state_feedback"]
        
        # # Create a records namespace with the proper mappings
        # records = SimpleNamespace()
        # records.u = motor_output            # mapping motor_output to records.u
        # records.u_lp = motor_output         # mapping motor_output to records.u_lp
        # records.state_feedback = state_feedback  # dedicated records attribute

        # Extract the first run from the simulation output
        runs = simulation_output["runs"]
        motor_output = np.nanmean([run["motor_output"] for _,run in runs.items()], axis=0)
        state_feedback = np.nanmean([run["state_feedback"] for _,run in runs.items()], axis=0)
        
        # Create a records namespace with the proper mappings
        records = SimpleNamespace()
        records.u = motor_output            # mapping motor_output to records.u
        records.u_lp = motor_output         # mapping motor_output to records.u_lp
        records.state_feedback = state_feedback  # dedicated records attribute
        records.full_output = simulation_output  # Store the full simulation output for debugging

        # Wrap the records inside an output object so that output.records can be accessed as expected
        output = SimpleNamespace(records=records)
        return output



def read_COIN_params(path, key_name='participant', relative_path=True):
    """
    at the path two files are expected:
    COIN_params_table.csv
    param_naming_map.csv

    return a dictionary of dictionaries where the keys are the names of the participants
    and the values are dictionaries with the parameters for each participant.
    The keys of the inner dictionaries are renamed according to the mapping file
    param_naming_map.csv if it exists.
    The mapping file has no header; columns are: old_name, new_name
    """
    if relative_path:
        path = COIN_path + path

    #reading the COIN_params_table.csv
    params = pd.read_csv(path + '/COIN_param_table.csv')
    #reading the param_naming_map.csv if exists
    try:
        param_map = pd.read_csv(path + '/param_naming_map.csv', header=None)
    except FileNotFoundError:
        param_map = None

    # mapping file has no header; columns are: old_name, new_name
    # replacing headers in the params table with the ones in the param_map, in both files trim spaces and backslashes from the names
    if param_map is not None:
        param_map.columns = ['old_name', 'new_name']
        param_map['old_name'] = param_map['old_name'].str.strip().str.replace('\\', '')
        param_map['new_name'] = param_map['new_name'].str.strip().str.replace('\\', '')
        params.columns = params.columns.str.strip().str.replace('\\', '')

        params.columns = [param_map[param_map['old_name'] == col]['new_name'].values[0] if col in param_map['old_name'].values else col for col in params.columns]

    # creating a dictionary of dictionaries where the keys are the names of the participants
    # and the values are dictionaries with the parameters for each participant.


    participants = {}
    for index, row in params.iterrows():
        participant = row[key_name]
   
        participants[participant] = {}
        for col in params.columns:
            if col != key_name:
                participants[participant][col] = row[col]

    # converting the values to floats
    # print('debug ddd',participants)
    # for participant in participants.keys():
    #     for key in participants[participant].keys():
    #         print('debug', participants[participant][key])
    #         participants[participant][key] = float(participants[participant][key])

    return participants


