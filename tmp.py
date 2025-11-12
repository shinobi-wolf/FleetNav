def __init__(self, config, render_mode=None): # Default render_mode is None
        super(SumoLMDEnv, self).__init__()

        '''LOADING THE CONFIGURATION VARIABLES'''
        self.net_path = config["net"]["net_path"]
        self.warehouse_choice = config["net"]["warehouse"]

        # Task vars
        self.demand_per = config['task']['demand_per']

        # Agent vars
        self.num_ugvs = config["fleet"]["count"]
        self.max_ugv_range = config["fleet"]['range']
        self.ugv_speed = config["fleet"]['max_speed']

        # Simulation vars
        self.max_time = config['simulation']['max_time']
        self.max_steps = config['simulation']['max_steps']
        self.render_mode = config['simulation']['render_mode']  # Render mode from config
        self.render_fps = config['simulation'].get('render_fps', 30)  # Default to 30 if not specified
        self.step_length = config['simulation']['step_length']
        self.ff = config['debug']['fast_forward']

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([4]*self.num_ugvs)
        
        ugv_status_space = spaces.MultiDiscrete([3]*self.num_ugvs) # 0: idle, 1: moving, 2: charging
        max_moves = 50
        steps2dest_space = spaces.MultiDiscrete(np.array([max_moves]*5*self.num_ugvs))
        self.observation_space = spaces.Dict({
            'ugv_status': ugv_status_space,
            'steps2dest': steps2dest_space,
        })