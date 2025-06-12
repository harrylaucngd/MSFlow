T_STEPS = 100
mask_sched = lambda t: t / T_STEPS  # linear corruption schedule for DM sampling