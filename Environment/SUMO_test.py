from gym import Env
from gym.spaces import Discrete, Box
from generator import TrafficGenerator, ModularTrafficGenerator
from utils import *
from State import *
from Statistics import *
from modular_road_network_structure import create_modular_road_network
from itertools import product
import random

config = import_train_configuration(config_file='training_settings.ini')
from Statistics_vehicles import Statistic_Vehicles
if not config['is_train']:
    config = import_test_configuration(config_file_path=config['test_model_path_name'])


class SUMO(Env):
    def __init__(self, stats):
        self.vehicle_stats = stats
        self.model_path, self.model_id = create_modular_road_network(config['models_path_name'], int(config['num_intersections']), int(config['intersection_length']))

        TrafficGen = ModularTrafficGenerator(
            config['max_steps'],
            config['n_cars_generated'],
            f'intersection/{self.model_path}/model_{self.model_id}/environment.net.xml'
        )

        self.TL_list, self.action_dict, self.program_dict, self.num_program_dict = self.get_tl_dicts(int(config['num_intersections']))

        self._TrafficGen = TrafficGen
        self._sumo_cmd=set_sumo(config['gui'], self.model_path, self.model_id, config['sumocfg_file_name'], config['max_steps'])

        if config['single_agent']:
            if config['fixed_action_space']:
                self.action_space = Discrete(config['num_actions'])
            else:
                self.single_action_space = Discrete(config['num_actions'])
                self.action_space = Discrete(pow(config['num_actions'], int(config['num_intersections'])))
                self.action_space_combinations = list(product(list(range(0, self.single_action_space.n)), repeat=int(config['num_intersections'])))
        else:
            self.action_space = Discrete(config['num_actions'])
            self.num_agents = len(self.TL_list)

        self.green_duration = config['green_duration']
        self.yellow_duration = config['yellow_duration']
        self.red_duration = config['red_duration']


        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self._TrafficGen.generate_routefile(model_path=self.model_path, model_id=self.model_id, seed=random.randint(0, 100))
        traci.start(self._sumo_cmd)

        self.junction_dict = init_states(self.TL_list)
        self.num_states = sum(i + 1 for i in list(get_state_size(self.junction_dict, self.num_program_dict).values()))
        self.init_state = [0] * self.num_states
        self.state = [0] * self.num_states

        self.junction_statistics_dict = init_statistics(self.TL_list)
        self.waiting_time = dict.fromkeys(TL_list)
        self.queue = dict.fromkeys(TL_list)

        self._old_action = dict.fromkeys(self.TL_list, 0)
        self._old_total_wait = 0
        self._waiting_times = {}
        self._waiting_old = 0
        self.reward = 0
        self.reward_old = 0



    def step(self, action):
        if config['fixed_action_space'] or config['single_agent'] is False:
            for tl_id in action:
                if self._old_action[tl_id] != action[tl_id]:
                    self._set_yellow_phase(self._old_action[tl_id], tl_id)
                else:
                    self._set_green_phase(action[tl_id], tl_id)
            self._simulate(self.yellow_duration)

            for tl_id in action:
                if self._old_action[tl_id] != action[tl_id]:
                    self._set_red_phase(self._old_action[tl_id], tl_id)
            self._simulate(self.red_duration)

            for tl_id in action:
                self._set_green_phase(action[tl_id], tl_id)
                self._simulate(self.green_duration)
                self._old_action[tl_id] = action[tl_id]
        else:
            action_combination = self.action_space_combinations[action]

            for tl_index, tl_id in enumerate(self.TL_list):
                if self._old_action[tl_id] != action_combination[tl_index]:
                    self._set_yellow_phase(self._old_action[tl_id], tl_id)
                else:
                    self._set_green_phase(action_combination[tl_index], tl_id)
            self._simulate(self.yellow_duration)

            for tl_index, tl_id in enumerate(self.TL_list):
                if self._old_action[tl_id] != action_combination[tl_index]:
                    self._set_red_phase(self._old_action[tl_id], tl_id)
            self._simulate(self.red_duration)

            for tl_index, tl_id in enumerate(self.TL_list):
                self._set_green_phase(action_combination[tl_index], tl_id)
                self._simulate(self.green_duration)
                self._old_action[tl_id] = action_combination[tl_index]

        # Removed because it slows training down
        # add_statistics_step(self.junction_statistics_dict)

        #get_states(self.junction_dict)

        if config['reward_definition'] == 'waiting':
            get_current_reward(self.junction_dict)
        # full_reward = return_reward(self.action_dict, self.junction_dict)
        # self.reward = sum(list(full_reward.values()))
        self.reward = return_reward(self.action_dict, self.junction_dict)
        self.reward_old = self.reward

        full_state = return_states(self.action_dict, self.junction_dict, self.program_dict, self.num_program_dict)
        self.state = [item for sublist in list(full_state.values()) for item in sublist]
        if len(self.state) == 0:
            self.state = self.init_state
        if traci.simulation.getTime() >= config['max_steps']+1000:
            done = True
            traci.close()
        else:
            done = False

        info = {}
        
        # Return step information
        return self.state, self.reward, done, info
    
    def _simulate(self, steps_todo):
            while steps_todo > 0:
                traci.simulationStep()
                # Removed because it slows training down
                # add_statistics_simulate(self.junction_statistics_dict)

                if config['reward_definition'] == 'waiting_fast':
                    get_current_reward(self.junction_dict)
                self.vehicle_stats.add_stats()
                get_states(self.junction_dict)
                steps_todo -= 1


    def _set_yellow_phase(self, old_action, tl_id):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 3 + 1  # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, yellow_phase_code)

    def _set_green_phase(self, action_number, tl_id):
        """
        Activate the correct green light combination in sumo
        """
        green_phase_code = action_number * 3  # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, green_phase_code)

    def _set_red_phase(self, old_action, tl_id):
        """
        Activate the correct red light combination in sumo
        """
        red_phase_code = old_action * 3 + 2  # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, red_phase_code)

    def reset(self):
        self._sumo_cmd = set_sumo(config['gui'], self.model_path, self.model_id, config['sumocfg_file_name'], config['max_steps'])
        # self._TrafficGen.generate_routefile(model_path=self.model_path, model_id=self.model_id, seed=random.randint(0, 9))
        try:
            traci.start(self._sumo_cmd)
        except:
            traci.close()
            traci.start(self._sumo_cmd)
        self.sim_length = config['max_steps']
        self.state = [0] * self.num_states
        self.waiting_time, self.queue = return_mean_statistics(self.waiting_time, self.queue, self.junction_statistics_dict)
        return self.state

    def render(self):
        # Implement viz
        pass

    def get_tl_dicts(self, num_intersections):
        TL_list = {}
        action_dict = {}
        program_dict = {}
        num_program_dict = {}

        for intersection in range(1, num_intersections + 1):
            TL_list[f'TL{intersection}'] = f'{intersection}'
            action_dict[f'TL{intersection}'] = 1
            program_dict[f'TL{intersection}'] = 0
            num_program_dict[f'TL{intersection}'] = 0

        return TL_list, action_dict, program_dict, num_program_dict