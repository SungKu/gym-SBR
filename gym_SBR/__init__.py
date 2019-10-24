from gym.envs.registration import register

register(id='SBR-v0',entry_point='gym_SBR.envs:SbrEnv',)
register(id='SBR-v1',entry_point='gym_SBR.envs:SbrEnv1',)
