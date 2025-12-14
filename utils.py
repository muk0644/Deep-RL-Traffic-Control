import os
import sys
import configparser
from sumolib import checkBinary
import pandas as pd
import datetime

import torch
from torch.autograd import Variable


def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['red_duration'] = content['simulation'].getint('red_duration')
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['policy_learning_rate'] = content['model'].getfloat('policy_learning_rate')
    config['value_learning_rate'] = content['model'].getfloat('value_learning_rate')
    config['actor_init_w'] = content['model'].getfloat('actor_init_w')
    config['critic_init_w'] = content['model'].getfloat('critic_init_w')
    config['weight_decay'] = content['model'].getfloat('weight_decay')
    config['training_epochs'] = content['model'].getint('training_epochs')
    config['target_update'] = content['model'].getint('target_update')
    config['warmup'] = content['model'].getint('warmup')
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')
    config['eps_start'] = content['strategy'].getfloat('eps_start')
    config['eps_end'] = content['strategy'].getfloat('eps_end')
    config['eps_decay'] = content['strategy'].getfloat('eps_decay')
    config['eps_policy'] = content['strategy'].getint('eps_policy')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['single_state_space'] = False if 'MA' not in content['agent']['agent_type'] else content['agent'].getboolean('single_state_space')
    config['fixed_action_space'] = False if 'MA' in content['agent']['agent_type'] else content['agent'].getboolean('fixed_action_space')
    config['local_reward_signal'] = False if 'MA' not in content['agent']['agent_type'] else content['agent'].getboolean('local_reward_signal')
    config['gamma'] = content['agent'].getfloat('gamma')
    print(type(content['model'].get('hidden_dim')))
    config['hidden_dim']=content['model'].get('hidden_dim').split(',')
    config['actor_dim'] = content['model'].get('actor_dim').split(',')
    config['critic_dim'] = content['model'].get('critic_dim').split(',')
    config['tau']=content['agent'].getfloat('tau')
    config['ou_theta']=content['agent'].getfloat('ou_theta')
    config['ou_mu']=content['agent'].getfloat('ou_mu')
    config['ou_sigma']=content['agent'].getfloat('ou_sigma')
    config['gae_lambda'] = content['agent'].getfloat('gae_lambda')
    config['policy_clip'] = content['agent'].getfloat('policy_clip')
    config['n_epochs'] = content['agent'].getint('n_epochs')
    config['models_path_name'] = content['dir']['models_path_name']
    config['test_model_path_name'] = content['dir']['test_model_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['generation_process'] = content['simulation']['generation_process']
    config['state_representation'] = content['agent']['state_representation']
    config['action_representation'] = content['agent']['action_representation']
    config['agent_type'] = content['agent']['agent_type']
    config['single_agent'] = False if 'MA' in config['agent_type'] else True
    config['model'] = content['agent']['model']
    config['is_train'] = content['agent'].getboolean('is_train')
    config['reward_definition'] = content['agent']['reward_definition']
    config['training_strategy'] = content['agent']['training_strategy']
    config['actor_parameter_sharing'] = content['agent'].getboolean('actor_parameter_sharing')
    config['critic_parameter_sharing'] = content['agent'].getboolean('critic_parameter_sharing')
    config['intersection_length'] = content['simulation']['intersection_length']
    config['num_intersections'] = content['simulation']['num_intersections']
    return config


def import_test_configuration(config_file_path):
    """
    Read the config file regarding the testing and import its content
    """
    config_file = os.path.join(config_file_path, 'training_settings.ini')
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['red_duration'] = content['simulation'].getint('red_duration')
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['policy_learning_rate'] = content['model'].getfloat('policy_learning_rate')
    config['value_learning_rate'] = content['model'].getfloat('value_learning_rate')
    config['actor_init_w'] = content['model'].getfloat('actor_init_w')
    config['critic_init_w'] = content['model'].getfloat('critic_init_w')
    config['weight_decay'] = content['model'].getfloat('weight_decay')
    config['training_epochs'] = content['model'].getint('training_epochs')
    config['target_update'] = content['model'].getint('target_update')
    config['warmup'] = content['model'].getint('warmup')
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')
    config['eps_start'] = content['strategy'].getfloat('eps_start')
    config['eps_end'] = content['strategy'].getfloat('eps_end')
    config['eps_decay'] = content['strategy'].getfloat('eps_decay')
    config['eps_policy'] = content['strategy'].getint('eps_policy')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['single_state_space'] = False if 'MA' not in content['agent']['agent_type'] else content['agent'].getboolean('single_state_space')
    config['fixed_action_space'] = False if 'MA' in content['agent']['agent_type'] else content['agent'].getboolean('fixed_action_space')
    config['local_reward_signal'] = False if 'MA' not in content['agent']['agent_type'] else content['agent'].getboolean('local_reward_signal')
    config['gamma'] = content['agent'].getfloat('gamma')
    print(type(content['model'].get('hidden_dim')))
    config['hidden_dim']=content['model'].get('hidden_dim').split(',')
    config['actor_dim'] = content['model'].get('actor_dim').split(',')
    config['critic_dim'] = content['model'].get('critic_dim').split(',')
    config['tau']=content['agent'].getfloat('tau')
    config['ou_theta']=content['agent'].getfloat('ou_theta')
    config['ou_mu']=content['agent'].getfloat('ou_mu')
    config['ou_sigma']=content['agent'].getfloat('ou_sigma')
    config['gae_lambda'] = content['agent'].getfloat('gae_lambda')
    config['policy_clip'] = content['agent'].getfloat('policy_clip')
    config['n_epochs'] = content['agent'].getint('n_epochs')
    config['models_path_name'] = content['dir']['models_path_name']
    config['test_model_path_name'] = config_file_path
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['generation_process'] = content['simulation']['generation_process']
    config['state_representation'] = content['agent']['state_representation']
    config['action_representation'] = content['agent']['action_representation']
    config['agent_type'] = content['agent']['agent_type']
    config['single_agent'] = False if 'MA' in config['agent_type'] else True
    config['model'] = content['agent']['model']
    config['is_train'] = False
    config['reward_definition'] = content['agent']['reward_definition']
    config['training_strategy'] = content['agent']['training_strategy']
    config['actor_parameter_sharing'] = content['agent'].getboolean('actor_parameter_sharing')
    config['critic_parameter_sharing'] = content['agent'].getboolean('critic_parameter_sharing')
    config['intersection_length'] = content['simulation']['intersection_length']
    config['num_intersections'] = content['simulation']['num_intersections']
    return config


def set_sumo(gui, model_path, model_id, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join(f'intersection/{model_path}/model_{model_id}', sumocfg_file_name), "--no-step-log", "true",
                "--waiting-time-memory", str(max_steps)]

    return sumo_cmd


def set_intersection_path(models_path_name):
    """
    Create a new intersection model path with an incremental integer, also considering previously created model paths
    """
    models_path = models_path_name.split('/', 1)[1]
    train_model_path = os.path.join(os.getcwd(), f'models/{models_path}', '')
    intersection_model_path = os.path.join(os.getcwd(), f'intersection/{models_path}', '')
    os.makedirs(os.path.dirname(train_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(intersection_model_path), exist_ok=True)

    dir_content = os.listdir(train_model_path)
    if dir_content:
        for d in dir_content:
            if d == '.DS_Store':
                os.remove(os.path.join(train_model_path, d))
                dir_content = os.listdir(train_model_path)
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(intersection_model_path, 'model_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return new_version


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        for d in dir_content:
            if d == '.DS_Store':
                os.remove(os.path.join(models_path, d))
                dir_content = os.listdir(models_path)
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(test_model_path_name):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), test_model_path_name, '')

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')


def add_masterdata(path, config, scores, training_time, wait, queue):
    master_df = pd.read_excel('Masterdata.xlsx')
    path = path[0:-1]
    name = os.path.split(path)[1]
    master_df = master_df.append({'run_name':name,
                      'datetime':datetime.datetime.now(),
                      'agent_type':config['agent_type'],
                      'model':config['model'],
                      'total_episodes':config['total_episodes'],
                      'generation_process':config['agent_type'],
                      'num_states':config['num_states'],
                      'cars_generated':config['n_cars_generated'],
                      'num_actions':config['num_actions'],
                       'state_representation':config['state_representation'],
                      'action_representation':config['action_representation'],
                      'final_reward':scores[-1],
                      'training_time':training_time[-1],
                    #   'final_waiting_time':wait[-1],
                    #   'final_length':queue[-1],
                      'final_waiting_time':wait,
                      'final_length':queue}, ignore_index=True)
    master_df.to_excel('Masterdata.xlsx', index=False)
