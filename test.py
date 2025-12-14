import os

import configparser
from subprocess import run


def test_models(test_path):
    test_models_path = test_path
    models_path = os.path.join(os.getcwd(), f'{test_models_path}', '')

    dir_content = os.listdir(models_path)
    dir_content.sort()
    if '.DS_Store' in dir_content:
        print('REMOVED: .DS_Store')
        dir_content.remove('.DS_Store')

    for model in dir_content:
        if 'ppofixed_5i_32_64' in model:
            config = configparser.ConfigParser()
            config.read('training_settings.ini')
            config.set('dir', 'test_model_path_name', f'{test_models_path}/{model}')

            with open('training_settings.ini', 'w') as configfile:
                config.write(configfile)

            run(["python3", "train.py"])


if __name__ == '__main__':
    test_models_single = ['models_ppofixed']

    test_models_fas = ['models_dqnfixed', 'models_ppofixed']
    test_models_wolp = ['models_wolp']
    test_models_madqn = [
        'models_madqnsinglelocal', 'models_madqnsingleglobal',
        'models_madqntotallocal', 'models_madqntotalglobal'
    ]
    test_models_mapposingle = [
        'models_mapposinglelocal/models_mapposinglelocal_nons',
        'models_mapposinglelocal/models_mapposinglelocal_conc',
        'models_mapposinglelocal/models_mapposinglelocal_cent',
        'models_mapposingleglobal/models_mapposingleglobal_nons',
        'models_mapposingleglobal/models_mapposingleglobal_conc',
        'models_mapposingleglobal/models_mapposingleglobal_cent'
    ]
    test_models_mappototal = [
        'models_mappototallocal/models_mappototallocal_nons',
        'models_mappototallocal/models_mappototallocal_conc',
        'models_mappototalglobal/models_mappototalglobal_nons',
        'models_mappototalglobal/models_mappototalglobal_conc'
    ]

    test_models_list = test_models_single

    test_models_complete_list = [
        'models_ppofixed',
        'models_madqnsinglelocal', 'models_madqnsingleglobal',
        'models_madqntotallocal', 'models_madqntotalglobal',
        'models_mapposinglelocal/models_mapposinglelocal_nons',
        'models_mapposinglelocal/models_mapposinglelocal_conc',
        'models_mapposinglelocal/models_mapposinglelocal_cent',
        'models_mapposingleglobal/models_mapposingleglobal_nons',
        'models_mapposingleglobal/models_mapposingleglobal_conc',
        'models_mapposingleglobal/models_mapposingleglobal_cent',
        'models_mappototallocal/models_mappototallocal_nons',
        'models_mappototallocal/models_mappototallocal_conc',
        'models_mappototalglobal/models_mappototalglobal_nons',
        'models_mappototalglobal/models_mappototalglobal_conc',
        'models_wolp'
    ]

    for model_to_test in test_models_list:
        test_models(test_path=f'models_test/{model_to_test}')


    # todo:
    # models_mapposinglelocal/models_mapposinglelocal_nons (5i)
    # models_mappototallocal/models_mappototallocal_nons (5i)
    # models_wolp

    # done:
    # models_dqn
    # models_ppo
    # models_dqnfixed
    # models_ppofixed
    # 
    # models_madqnsinglelocal
    # models_madqnsingleglobal
    # models_madqntotallocal
    # models_madqntotalglobal
    # 
    # models_mapposinglelocal/models_mapposinglelocal_nons (2-4i)
    # models_mapposinglelocal/models_mapposinglelocal_conc
    # models_mapposinglelocal/models_mapposinglelocal_cent
    # models_mapposingleglobal/models_mapposingleglobal_nons
    # models_mapposingleglobal/models_mapposingleglobal_conc
    # models_mapposingleglobal/models_mapposingleglobal_cent
    # 
    # models_mappototallocal/models_mappototallocal_nons
    # models_mappototallocal/models_mappototallocal_conc
    # models_mappototalglobal/models_mappototalglobal_nons
    # models_mappototalglobal/models_mappototalglobal_conc
    # 

    # todo:
    # 
    # 7045:
    # models_madqnsinglelocal
    # models_madqnsingleglobal
    # models_madqntotallocal
    # models_madqntotalglobal
    # 
    # todo:
    # 
    # 7030:
    # models_mapposinglelocal/models_mapposinglelocal_nons
    # models_mapposinglelocal/models_mapposinglelocal_conc
    # models_mapposinglelocal/models_mapposinglelocal_cent
    # models_mapposingleglobal/models_mapposingleglobal_nons
    # models_mapposingleglobal/models_mapposingleglobal_conc
    # models_mapposingleglobal/models_mapposingleglobal_cent
    # 
    # todo:
    # 
    # 7055:
    # models_mappototallocal/models_mappototallocal_nons
    # models_mappototallocal/models_mappototallocal_conc
    # models_mappototalglobal/models_mappototalglobal_nons
    # models_mappototalglobal/models_mappototalglobal_conc
    # 
    # todo:
    # 



# for i in range(10):
#     config=configparser.ConfigParser()
#     config.read('training_settings.ini')
#     config.set('simulation', 'num_intersections', str(i+1))

#     with open('training_settings.ini', 'w') as configfile:
#         config.write(configfile)


#     call(["python", "train.py"])