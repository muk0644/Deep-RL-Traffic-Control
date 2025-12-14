import os
import glob
import math
from operator import sub

from xml.dom import minidom
from shutil import copy

from utils import set_intersection_path


def create_node_xmlfile(model_path, model_id, num_nodes, len_edges):
    def get_grid_size(num_nodes):
        grid_base = math.floor(math.sqrt(num_nodes))
        grid_base_nodes = int(math.pow(grid_base, 2))
        grid_extra_nodes = num_nodes - grid_base_nodes

        return grid_base, grid_extra_nodes

    xml = minidom.Document()

    nodes = xml.createElement('nodes')
    xml.appendChild(nodes)

    size_generic_grid, num_extra_nodes = get_grid_size(num_nodes=num_nodes)
    square_grid = {}
    grid_size_counter = 1

    def update_square_grid(coordinate_id, coordinate_value):
        nonlocal grid_size_counter

        square_grid[coordinate_id] = coordinate_value
        grid_size_counter += 1

    """Generate square grid for nodes with coordinates"""
    if num_nodes < 4:
        if num_nodes == 1:
            update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(1, 1))
        elif num_nodes == 2:
            update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(1, 1))
            update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(1, 2))
        elif num_nodes == 3:
            update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(1, 1))
            update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(1, 2))
            update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(2, 2))
    else:
        for grid_base in range(size_generic_grid - 1):
            if grid_base == 0:
                update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(1, 1))
                update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(1, 2))
                update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(2, 2))
                update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(2, 1))
            else:
                coordinate_start_value = 1
                coordinate_end_value = grid_base + 2
                loop_range = 3 + 2 * grid_base
                loop_halfway_point = math.floor(loop_range / 2)

                for grid_node in range(loop_range):
                    if grid_node < loop_halfway_point:
                        update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(coordinate_start_value, coordinate_end_value))
                        coordinate_start_value += 1
                    else:
                        update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(coordinate_end_value, coordinate_start_value))
                        coordinate_start_value -= 1
        
        loop_range_extra = 3 + 2 * (size_generic_grid - 1)
        loop_halfway_point_extra = math.floor(loop_range_extra / 2)
        coordinate_extra_start_value = 1
        coordinate_extra_end_value = square_grid[f'{grid_size_counter - 1}'][0] + 1

        for grid_extra in range(num_extra_nodes):
            if grid_extra < loop_halfway_point_extra:
                update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(coordinate_extra_start_value, coordinate_extra_end_value))
                coordinate_extra_start_value += 1
            else:
                update_square_grid(coordinate_id=f'{grid_size_counter}', coordinate_value=(coordinate_extra_end_value, coordinate_extra_start_value))
                coordinate_extra_start_value -= 1

    """Add helper nodes for connection edges to outer nodes"""
    grid_coordinates = []
    grid_x_coordinates = []
    grid_y_coordinates = []

    for square_grid_node in square_grid:
        grid_coordinates.append(square_grid[f'{square_grid_node}'])
        grid_x_coordinates.append(square_grid[f'{square_grid_node}'][0])
        grid_y_coordinates.append(square_grid[f'{square_grid_node}'][1])

    square_grid_list = []
    square_grid_list_x = []
    square_grid_list_y = []
    square_grid_list_original = []
    square_grid_y = []
    for y_nodes in range(max(grid_y_coordinates) + 2):
        square_grid_y.append(None)

    for x_nodes in range(max(grid_x_coordinates) + 2):
        square_grid_list.append(list(square_grid_y))
        square_grid_list_x.append(list(square_grid_y))
        square_grid_list_y.append(list(square_grid_y))
        square_grid_list_original.append(list(square_grid_y))

    for node_coordinate in grid_coordinates:
        square_grid_list[node_coordinate[0]][node_coordinate[1]] = node_coordinate
        square_grid_list_x[node_coordinate[0]][node_coordinate[1]] = node_coordinate
        square_grid_list_y[node_coordinate[0]][node_coordinate[1]] = node_coordinate
        square_grid_list_original[node_coordinate[0]][node_coordinate[1]] = node_coordinate

    for list_x in range(len(square_grid_list) - 1):
        list_x_value_one = square_grid_list_original[list_x]
        list_x_value_two = square_grid_list_original[list_x + 1]

        for list_y in range(1, len(list_x_value_one) - 1):
            if list_x_value_one[list_y] is None and list_x_value_two[list_y] is not None:
                square_grid_list_x[list_x][list_y] = (list_x_value_two[list_y][0] - 0.75, list_x_value_two[list_y][1])
            elif list_x_value_one[list_y] is not None and list_x_value_two[list_y] is None:
                if list_x_value_two[list_y - 1] is None and list_x_value_two[list_y + 1] is None:
                    square_grid_list_x[list_x + 1][list_y] = (list_x_value_one[list_y][0] + 0.75, list_x_value_one[list_y][1])
                else:
                    square_grid_list_x[list_x + 1][list_y] = [(list_x_value_one[list_y][0] + 0.75, list_x_value_one[list_y][1]), None]

    for list_y in range(1, len(square_grid_list) - 1):
        list_x_value = square_grid_list_original[list_y]
        list_x_value_dimensions = [0, len(list_x_value) - 1]

        if list_x_value.count(None) == 2:
            for list_x in list_x_value_dimensions:
                edge_list_x = list_x + 0.25 if list_x == 0 else list_x - 0.25
                square_grid_list_y[list_y][list_x] = (list_y, edge_list_x)
        else:
            for list_x in range(list_x_value_dimensions[1], -1, -1):
                if list_x == 0 and square_grid_list_original[list_y][1] is not None:
                    square_grid_list_y[list_y][0] = (list_y, 0.25)
                elif square_grid_list_original[list_y][list_x] is None and square_grid_list_original[list_y][list_x - 1] is not None:
                    if square_grid_list_original[list_y - 1][list_x] is None:
                        square_grid_list_y[list_y][list_x] = (list_y, list_x - 0.25)
                    else:
                        square_grid_list_y[list_y][list_x] = [None, (list_y, list_x - 0.25)]
                elif square_grid_list_original[list_y][list_x] is not None and square_grid_list_original[list_y][list_x - 1] is None:
                    if square_grid_list_original[list_y - 1][list_x - 1] is None:
                        square_grid_list_y[list_y][list_x - 1] = (list_y, list_x - 0.75)
                    else:
                        square_grid_list_y[list_y][list_x - 1] = [None, (list_y, list_x - 0.75)]

    for list_x in range(len(square_grid_list)):
        for list_y in range(len(square_grid_list[list_x])):
            if not isinstance(square_grid_list_x[list_x][list_y], list) and not isinstance(square_grid_list_y[list_x][list_y], list):
                if square_grid_list_x[list_x][list_y] is None and square_grid_list_y[list_x][list_y] is not None:
                    square_grid_list[list_x][list_y] = square_grid_list_y[list_x][list_y]
                elif square_grid_list_x[list_x][list_y] is not None and square_grid_list_y[list_x][list_y] is None:
                    square_grid_list[list_x][list_y] = square_grid_list_x[list_x][list_y]
            elif isinstance(square_grid_list_x[list_x][list_y], list) and isinstance(square_grid_list_y[list_x][list_y], list):
                square_grid_list[list_x][list_y] = [square_grid_list_x[list_x][list_y][0], square_grid_list_y[list_x][list_y][1]]
            else:
                if square_grid_list_x[list_x][list_y] is None and square_grid_list_y[list_x][list_y] is not None:
                    square_grid_list[list_x][list_y] = square_grid_list_y[list_x][list_y][1]
                elif square_grid_list_x[list_x][list_y] is not None and square_grid_list_y[list_x][list_y] is None:
                    square_grid_list[list_x][list_y] = square_grid_list_x[list_x][list_y][0]

    for list_y in square_grid_list:
        for list_x in list_y:
            if list_x is not None and list_x not in grid_coordinates:
                for grid_key_index in range(1, 3 if isinstance(list_x, list) else 2):
                    if 'c' in list(square_grid)[-1]:
                        square_grid_key = list(square_grid)[-1].replace('c', '')
                        grid_key = f'c{int(square_grid_key) + grid_key_index}'
                    else:
                        grid_key = f'c{int(list(square_grid)[-1]) + grid_key_index}'
                    square_grid[grid_key] = list_x[grid_key_index - 1] if isinstance(list_x, list) else list_x

    """Create nod.xml file"""
    for node_element in square_grid:
        node_x = square_grid[node_element][0] * len_edges
        node_y = square_grid[node_element][1] * len_edges

        nodeChild = xml.createElement('node')
        nodeChild.setAttribute('id', node_element)
        nodeChild.setAttribute('x', f'{node_x}')
        nodeChild.setAttribute('y', f'{node_y}')
        nodeChild.setAttribute('type', 'unregulated' if 'c' in node_element else 'traffic_light')
        if not 'c' in node_element:
            nodeChild.setAttribute('tl', f'TL{node_element}')

        nodes.appendChild(nodeChild)

    xml_str = xml.toprettyxml(indent ="\t")

    node_file_path = f'intersection/{model_path}/model_{model_id}/Ingolstadt_{num_nodes}_Nodes.nod.xml'

    with open(node_file_path, "w") as f:
        f.write(xml_str)

    return square_grid, square_grid_list, square_grid_list_original, node_file_path

def create_edge_xmlfile(model_path, model_id, square_grid, square_grid_list, square_grid_list_original):
    xml = minidom.Document()

    edges = xml.createElement('edges')
    xml.appendChild(edges)

    square_grid_edges = {}
    grid_edge_size_counter = 1

    def update_square_grid_edges(coordinate_id, coordinate_value):
        nonlocal grid_edge_size_counter

        square_grid_edges[coordinate_id] = coordinate_value
        grid_edge_size_counter += 1

    """Add edges to inner nodes"""
    grid_coordinates = []
    square_grid_nodes = []
    square_grid_edge_nodes = []

    for square_grid_node in square_grid:
        grid_coordinates.append(square_grid[f'{square_grid_node}'])
        if 'c' in square_grid_node:
            square_grid_edge_nodes.append(square_grid[f'{square_grid_node}'])
        else:
            square_grid_nodes.append(square_grid[f'{square_grid_node}'])

    edge_combinations = []
    for list_x in range(1, len(square_grid_list_original) - 1):
        list_x_start = square_grid_list_original[list_x]
        list_x_end = square_grid_list_original[list_x + 1]

        for list_y in range(1, len(list_x_start) - 1):
            if list_x_start[list_y] is not None and list_x_end[list_y] is not None:
                edge_combinations.append([list_x_start[list_y], list_x_end[list_y]])
                edge_combinations.append([list_x_end[list_y], list_x_start[list_y]])
            if list_x_start[list_y] is not None and list_x_start[list_y + 1] is not None:
                edge_combinations.append([list_x_start[list_y], list_x_start[list_y + 1]])
                edge_combinations.append([list_x_start[list_y + 1], list_x_start[list_y]])

    for grid_edge in edge_combinations:
        edge_from = list(square_grid.keys())[list(square_grid.values()).index(grid_edge[0])]
        edge_to = list(square_grid.keys())[list(square_grid.values()).index(grid_edge[1])]
        update_square_grid_edges(coordinate_id=f'{grid_edge_size_counter}', coordinate_value=(edge_from, edge_to))

    """Add connection edges to outer nodes"""
    edge_connection_combinations = []
    for list_x in range(0, len(square_grid_list) - 1):
        list_x_start = square_grid_list[list_x]
        list_x_end = square_grid_list[list_x + 1]
        
        for list_y in range(0, len(list_x_start) - 1):
            if isinstance(list_x_start[list_y], list) or isinstance(list_x_end[list_y], list) or isinstance(list_x_start[list_y + 1], list):
                list_x_start_list_y = list_x_start[list_y][1] if isinstance(list_x_start[list_y], list) else list_x_start[list_y]
                list_x_end_list_y = list_x_end[list_y][0] if isinstance(list_x_end[list_y], list) else list_x_end[list_y]
                list_x_start_list_y_1 = list_x_start[list_y + 1][1] if isinstance(list_x_start[list_y + 1], list) else list_x_start[list_y + 1]
            else:
                list_x_start_list_y = list_x_start[list_y]
                list_x_end_list_y = list_x_end[list_y]
                list_x_start_list_y_1 = list_x_start[list_y + 1]

            if list_x_start_list_y not in square_grid_edge_nodes or list_x_end_list_y not in square_grid_edge_nodes:
                if list_x_start_list_y is not None and list_x_end_list_y is not None:
                    if list_x_start_list_y[0] == list_x_end_list_y[0] or list_x_start_list_y[1] == list_x_end_list_y[1]:
                        edge_connection_combinations.append([list_x_start_list_y, list_x_end_list_y])
                        edge_connection_combinations.append([list_x_end_list_y, list_x_start_list_y])

            if list_x_start_list_y not in square_grid_edge_nodes or list_x_start_list_y_1 not in square_grid_edge_nodes:
                if list_x_start_list_y is not None and list_x_start_list_y_1 is not None:
                    if list_x_start_list_y[0] == list_x_start_list_y_1[0] or list_x_start_list_y[1] == list_x_start_list_y_1[1]:
                        edge_connection_combinations.append([list_x_start_list_y, list_x_start_list_y_1])
                        edge_connection_combinations.append([list_x_start_list_y_1, list_x_start_list_y])

    edge_connections = [x for x in edge_connection_combinations if x not in edge_combinations]

    for connection_edge in edge_connections:
        edge_from = list(square_grid.keys())[list(square_grid.values()).index(connection_edge[0])]
        edge_to = list(square_grid.keys())[list(square_grid.values()).index(connection_edge[1])]
        update_square_grid_edges(coordinate_id=f'{grid_edge_size_counter}', coordinate_value=(edge_from, edge_to))

    """Create edg.xml file"""
    for edge_element in range(len(square_grid_edges)):
        edge_id = f'ce{edge_element + 1}' if 'c' in square_grid_edges[f'{edge_element + 1}'][0] or 'c' in square_grid_edges[f'{edge_element + 1}'][1] else f'e{edge_element + 1}'
        edge_from = square_grid_edges[f'{edge_element + 1}'][0]
        edge_to = square_grid_edges[f'{edge_element + 1}'][1]

        edgeChild = xml.createElement('edge')
        edgeChild.setAttribute('id', edge_id)
        edgeChild.setAttribute('from', edge_from)
        edgeChild.setAttribute('to', edge_to)
        edgeChild.setAttribute('numLanes', '4')
        edgeChild.setAttribute('speed', '13.9')

        edges.appendChild(edgeChild)

    xml_str = xml.toprettyxml(indent ="\t")

    edge_file_path = f'intersection/{model_path}/model_{model_id}/Ingolstadt_{int(len(edge_combinations) / 2)}_Edges.edg.xml'

    with open(edge_file_path, "w") as f:
        f.write(xml_str)
    
    return square_grid_edges, edge_file_path

def create_connection_xmlfile(model_path, model_id, square_grid, square_grid_edges):
    xml = minidom.Document()

    connections = xml.createElement('connections')
    xml.appendChild(connections)

    def create_connection_child(edge_from, edge_to, lane_from, lane_to):
        nonlocal connections

        connectionChild = xml.createElement('connection')
        connectionChild.setAttribute('from', edge_from)
        connectionChild.setAttribute('to', edge_to)
        connectionChild.setAttribute('fromLane', lane_from)
        connectionChild.setAttribute('toLane', lane_to)
        connections.appendChild(connectionChild)

    """Define lane specific connections for edges"""
    square_grid_edges_list = list(square_grid_edges.values())

    connection_combinations = []
    
    for list_x in range(1, len(square_grid_edges) + 1):
        node_edge = square_grid_edges[f'{list_x}']
        connection_edges = [x for x in square_grid_edges_list if x[0] == node_edge[1] and x[1] != node_edge[0]]

        node_from_to = tuple(map(sub, square_grid[node_edge[1]], square_grid[node_edge[0]]))

        if node_from_to[1] == 0:
            node_direction = '+x' if node_from_to[0] == 1 or node_from_to[0] == 0.75 else '-x'
        elif node_from_to[0] == 0:
            node_direction = '+y' if node_from_to[1] == 1 or node_from_to[1] == 0.75 else '-y'

        if connection_edges:
            for edge in connection_edges:
                edge_node_from = square_grid[edge[0]]
                edge_node_to = square_grid[edge[1]]
                edge_node_from_to = tuple(map(sub, edge_node_to, edge_node_from))

                if ('x' in node_direction and edge_node_from_to[1] == 0) or ('y' in node_direction and edge_node_from_to[0] == 0):
                    edge_direction = 's'
                elif 'x' in node_direction and edge_node_from_to[0] == 0:
                    if node_direction == '+x':
                        edge_direction = 'l' if edge_node_from_to[1] == 1 or edge_node_from_to[1] == 0.75 else 'r'
                    elif node_direction == '-x':
                        edge_direction = 'r' if edge_node_from_to[1] == 1 or edge_node_from_to[1] == 0.75 else 'l'
                elif 'y' in node_direction and edge_node_from_to[1] == 0:
                    if node_direction == '+y':
                        edge_direction = 'r' if edge_node_from_to[0] == 1 or edge_node_from_to[0] == 0.75 else 'l'
                    elif node_direction == '-y':
                        edge_direction = 'l' if edge_node_from_to[0] == 1 or edge_node_from_to[0] == 0.75 else 'r'

                edge_from = f'ce{list_x}' if 'c' in node_edge[0] or 'c' in node_edge[1] else f'e{list_x}'
                edge_id = list(square_grid_edges.keys())[list(square_grid_edges.values()).index(edge)]
                edge_to = f'ce{edge_id}' if 'c' in edge[0] or 'c' in edge[1] else f'e{edge_id}'

                connection_combinations.append({edge_from: [edge_to, edge_direction]})

    connection_combination_dict = {}
    for connection_combination in connection_combinations:
        connection_combination_key = list(connection_combination.keys())[0]
        connection_combination_direction = connection_combination[connection_combination_key][1]

        if connection_combination_key not in connection_combination_dict:
            connection_combination_dict[connection_combination_key] = [connection_combination_direction]
        if connection_combination_direction not in connection_combination_dict[connection_combination_key]:
            connection_combination_dict[connection_combination_key].append(connection_combination_direction)

    """Create con.xml file"""
    for connection_element in connection_combinations:
        edge_from = list(connection_element.keys())[0]
        edge_to = connection_element[edge_from][0]
        edge_direction = connection_element[edge_from][1]

        if edge_direction == 's':
            for lane in range(0, 3):
                create_connection_child(edge_from=edge_from, edge_to=edge_to, lane_from=f'{lane}', lane_to=f'{lane}')
        else:
            lane_from_to = '0' if edge_direction == 'r' else '3'
            create_connection_child(edge_from=edge_from, edge_to=edge_to, lane_from=lane_from_to, lane_to=lane_from_to)

    xml_str = xml.toprettyxml(indent ="\t")

    connection_file_path = f'intersection/{model_path}/model_{model_id}/Ingolstadt_{len(square_grid_edges)}_Connections.con.xml'

    with open(connection_file_path, "w") as f:
        f.write(xml_str)
    
    return connection_file_path

def create_tllogic_xmlfile(model_path, model_id, square_grid):
    xml = minidom.Document()

    tllogics = xml.createElement('tlLogics')
    xml.appendChild(tllogics)

    square_grid_nodes_list = [x for x in list(square_grid.keys()) if 'c' not in x]

    phase_xml_states = [
        'GGGGrrrrrrGGGGrrrrrr', 'yyyyrrrrrryyyyrrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrGrrrrrrrrrGrrrrr', 'rrrryrrrrrrrrryrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrGGGGrrrrrrGGGGr', 'rrrrryyyyrrrrrryyyyr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrrrrrGrrrrrrrrrG', 'rrrrrrrrryrrrrrrrrry', 'rrrrrrrrrrrrrrrrrrrr',
        'GGGGGrrrrrrrrrrrrrrr', 'yyyyyrrrrrrrrrrrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrGGGGGrrrrrrrrrr', 'rrrrryyyyyrrrrrrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrrrrrrGGGGGrrrrr', 'rrrrrrrrrryyyyyrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrrrrrrrrrrrGGGGG', 'rrrrrrrrrrrrrrryyyyy', 'rrrrrrrrrrrrrrrrrrrr',
    ]

    """Create tll.xml file"""
    for tllogic_element in square_grid_nodes_list:
        tllogic_id = f'TL{tllogic_element}'

        tlLogicChild = xml.createElement('tlLogic')
        tlLogicChild.setAttribute('id', tllogic_id)
        tlLogicChild.setAttribute('type', 'static')
        tlLogicChild.setAttribute('programID', '0')
        tlLogicChild.setAttribute('offset', '0')

        for phase_element in phase_xml_states:
            phaseChild = xml.createElement('phase')
            phaseChild.setAttribute('duration', '100')
            phaseChild.setAttribute('state', phase_element)
            tlLogicChild.appendChild(phaseChild)

        tllogics.appendChild(tlLogicChild)

    xml_str = xml.toprettyxml(indent ="\t")

    tllogic_file_path = f'intersection/{model_path}/model_{model_id}/Ingolstadt_{len(square_grid_nodes_list)}_TrafficLights.tll.xml'

    with open(tllogic_file_path, "w") as f:
        f.write(xml_str)
    
    return tllogic_file_path

def create_env_xmlfile(model_path, model_id, node_files, edge_files, connection_files, tllogic_files):
    env_file = f'intersection/{model_path}/model_{model_id}/environment.net.xml'

    os.system(f'netconvert --node-files={node_files} --edge-files={edge_files} --connection-files={connection_files} --tllogic-files={tllogic_files} -o {env_file}')

def create_modular_road_network(models_path_name, number_nodes, length_edges=100):
    model_id = set_intersection_path(models_path_name)
    model_path = models_path_name.split('/', 1)[1]

    # files = glob.glob('intersection/*')
    # for f in files:
    #     if f[-1]=="l" and f[-5]!="u":
    #         os.remove(f)

    square_grid, square_grid_list, square_grid_list_original, node_file = create_node_xmlfile(model_path=model_path, model_id=model_id, num_nodes=number_nodes, len_edges=length_edges)
    square_grid_edges, edge_file = create_edge_xmlfile(model_path=model_path, model_id=model_id, square_grid=square_grid, square_grid_list=square_grid_list, square_grid_list_original=square_grid_list_original)
    connection_file = create_connection_xmlfile(model_path=model_path, model_id=model_id, square_grid=square_grid, square_grid_edges=square_grid_edges)
    tllogic_file = create_tllogic_xmlfile(model_path=model_path, model_id=model_id, square_grid=square_grid)
    create_env_xmlfile(model_path=model_path, model_id=model_id, node_files=node_file, edge_files=edge_file, connection_files=connection_file, tllogic_files=tllogic_file)

    copy('intersection/sumo_config.sumocfg', f'intersection/{model_path}/model_{model_id}/sumo_config.sumocfg')

    return model_path, model_id


if __name__ == '__main__':
    create_modular_road_network('models', 50, 200)