from gym.envs.registration import register

register(id='SBR-v0',entry_point='gym_SBR.envs:SbrEnv',)
register(id='SBR-v1',entry_point='gym_SBR.envs:SbrEnv1',)
register(id='SBR-v2',entry_point='gym_SBR.envs:SbrEnv2',)
register(id='SBR-v4',entry_point='gym_SBR.envs:SbrEnv4',)
register(id='SBRCnt-v0',entry_point='gym_SBR.envs:SbrCnt0',)
