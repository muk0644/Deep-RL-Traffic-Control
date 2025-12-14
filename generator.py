import math
import numpy as np, numpy.random
from xml.dom import minidom
import random


class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/dummy/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)


class ModularTrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, netfile):
        self.netfile = netfile
        self.max_steps = max_steps
        self.n_cars_generated = n_cars_generated
        self.junction_dict = self.get_outer_junctions()
        #self.rout_list = self.generate_rout_dict(0)
        #print(self.rout_list)
        #self.generate_routfile(0)


    def get_outer_junctions(self) -> dict:
        xml_net = minidom.parse(self.netfile)
        junction_list = xml_net.getElementsByTagName('junction')
        junction_dict = {None:None}
        for j in junction_list:
            if j.attributes['id'].value[0] == "c":
                junction_dict[j.attributes['id'].value] = {'x': float(j.attributes['x'].value), 'y': float(j.attributes['y'].value), 'type': None}
        junction_dict.pop(None)
        max_x = max(float(d['x']) for d in junction_dict.values())
        min_x = min(float(d['x']) for d in junction_dict.values())
        max_y = max(float(d['y']) for d in junction_dict.values())
        min_y = min(float(d['y']) for d in junction_dict.values())

        for junction in junction_dict:
            if junction_dict[junction]['x'] == max_x:
                junction_dict[junction]['type'] = 'east'
            if junction_dict[junction]['x'] == min_x:
                junction_dict[junction]['type'] = 'west'
            if junction_dict[junction]['y'] == max_y:
                junction_dict[junction]['type'] = 'north'
            if junction_dict[junction]['y'] == min_y:
                junction_dict[junction]['type'] = 'south'

        edge_list = xml_net.getElementsByTagName('edge')
        for junction in junction_dict:
            if junction_dict[junction]['type'] is None:
                #print(junction)
                for e in edge_list:
                    if e.attributes['id'].value[0] != ':':
                        if e.attributes['to'].value == junction:
                            next_junction=e.attributes['from'].value
                            for j in junction_list:
                                if j.attributes['id'].value == next_junction:
                                    next_junction_x = float(j.attributes['x'].value)
                                    next_junction_y = float(j.attributes['y'].value)
                            if junction_dict[junction]['x'] > next_junction_x:
                                junction_dict[junction]['type'] = 'east'
                            if junction_dict[junction]['x'] < next_junction_x:
                                junction_dict[junction]['type'] = 'west'
                            if junction_dict[junction]['y'] > next_junction_y:
                                junction_dict[junction]['type'] = 'north'
                            if junction_dict[junction]['y'] < next_junction_y:
                                junction_dict[junction]['type'] = 'south'
        return junction_dict

    def generate_rout_dict(self,seed) -> list:
        xml_net = minidom.parse(self.netfile)
        """
        Generation of the route of every car for one episode.
        """
        direction_dict = {'south': [], 'north': [], 'west': [], 'east': []}
        for junction in self.junction_dict:
            direction_dict[self.junction_dict[junction]['type']].append(junction)
        #print(direction_dict)
        """
        We randomly specify if a car comes from the south, north, west or east.
        A car generates has a 30% probability to move straight or move to the left or right end of the system and
        a 10% probability to move to the same direction.
        We only consider transit traffic.
        First we need to create cars according to the Weibull distribution
        """
        np.random.seed(seed)  # make tests reproducible
        # timings = np.random.weibull(2, self.n_cars_generated)
        timings = np.random.uniform(0, self.max_steps, self.n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        # car_gen_steps = []
        # min_old = math.floor(timings[1])
        # max_old = math.ceil(timings[-1])
        # min_new = 0
        # max_new = self.max_steps
        # for value in timings:
        #     car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)
        car_gen_steps = timings
        car_gen_steps = np.rint(car_gen_steps)
        #print(car_gen_steps)
        """
        We now need to assign the cars to teh direction they come from
        """
        source_prob=np.random.dirichlet(np.ones(4),size=1)
        #print(source_prob)
        rout_list=list()
        counter=0
        source_list=['south', 'west', 'north', 'east']
        edge_list = xml_net.getElementsByTagName('edge')
        for depart in range(len(car_gen_steps)):
            source = random.choices(source_list, weights=source_prob[0], k=1)[0]
            source_junction=random.choices(direction_dict[source], weights=np.random.dirichlet(np.ones(len(direction_dict[source])),size=1)[0],k=1)[0]
            for e in edge_list:
                if e.attributes['id'].value[0] != ':':
                    if e.attributes['from'].value == source_junction:
                        source_edge = e.attributes['id'].value

            target_prob=list()
            for av_source in source_list:
                if source == av_source:
                    target_prob.append(0.1)
                else:
                    target_prob.append(0.3)

            target = random.choices(source_list, weights=target_prob, k=1)[0]
            target_junction=random.choices(direction_dict[target], weights=np.random.dirichlet(np.ones(len(direction_dict[target])),size=1)[0],k=1)[0]
            for e in edge_list:
                if e.attributes['id'].value[0] != ':':
                    if e.attributes['to'].value == target_junction:
                        target_edge = e.attributes['id'].value

            new_car=[car_gen_steps[depart], source_edge, target_edge]
            rout_list.append(new_car)

        return rout_list

    def generate_routefile(self, model_path, model_id, seed):
        rout_list = self.generate_rout_dict(seed)
        with open(f"intersection/{model_path}/model_{model_id}/modular_routes.rou.xml", "w") as routes:
            print("""<routes>
             <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
                    """, file=routes)

            for car_counter, car in enumerate(rout_list):
                print(
                    f'    <trip id="{car_counter}" depart="{car[0]}" from="{car[1]}" to="{car[2]}" type="standard_car" departLane="random" departSpeed="10" />'
                    , file=routes)
            print("</routes>", file=routes)


if __name__ == '__main__':
    generator=ModularTrafficGenerator(3600,1000,"Modular_Road_Network_Structure//intersection//Ingolstadt_Environment.net.xml")
