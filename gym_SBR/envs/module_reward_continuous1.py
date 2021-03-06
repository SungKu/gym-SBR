import numpy



def sbr_reward( x_out, u_t, done, eff):
    t_delta = 0.002 / 24
    snh = x_out[10]

    # ========= OCI ==========
    dt = t_delta

    T = 0.5  # 12hrs, 0.5 day

    # Mechanical Eergy (kWh / d)
    # Assume: anerobic phase에서 mixing
    # ME_2 = 0.005*1.32*24
    # ME_4 = 0.005*1.32*24
    # ME = ME_2 + ME_4

    if done:  # Settling, Drawing, idle phases
        Q_eff = eff[0]
        Snh = eff[3]
        Qw = eff[6]

        # Pumping energy (kWh / d)
        PE = (0.05 * Qw + 0.004 * Q_eff)  # SBR에는 내부 외부 반송 없음
        # Aeration energy (kWh / d)
        # AE_deltaT = 1.32*sum(Kla)#*t_delta
        r_e = ((0 - 0.5) / (8 - 0)) * (x_out[8] - 0) + 0.5
        r_n = ((0 - 0.5) / (4 - 0)) * (snh - 0) + 0.5

        if Snh < 4:
            r_Snh = 0
        else:
            r_Snh = -246

    else:  # Reaction phases

        # Aeration energy (kWh / d)
        r_e = ((0 - 1) / (8 - 0)) * (x_out[8] - 0) + 1  # AE_deltaT = 1.32*Kla[-1]#*t_delta
        r_n = ((0 - 1) / (4 - 0)) * (snh - 0) + 1

        # sum(kla_memory3)*t_delta/(len(kla_memory3)*t_delta)
        PE = 0

        r_Snh = 0

    # AE = So_sat / (1.8 * 1000) * (AE_deltaT)

    # OCI = AE # + PE  #+ ME #+ 5*SP

    # =============================

    # r_OCI = 0.5 - OCI

    reward = r_Snh + r_e + r_n  # + r_OCI

    return reward
