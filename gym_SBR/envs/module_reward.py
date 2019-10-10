import numpy


def sbr_reward(x_last,  DO_control_par, kla_memory_3, kla_memory_5, kla_memory_8, Qeff,Qw):
    """
     List of variables :
                0=V, 1=Si, 2=Ss, 3=Xi, 4=Xs, 5=Xbh, 6=Xba, 7=Xp, 8=So, 9=Sno, 10=Snh, 11=Snd, 12=Xnd, 13=Salk
                (ref. BSM1 report Tbl. 1)
    """

    # 데이터 불러오기
    V = x_last[0]
    Si = x_last[1]
    Ss = x_last[2]
    Xi = x_last[3]
    Xs = x_last[4]
    Xbh = x_last[5]
    Xba = x_last[6]
    Xp = x_last[7]
    So = x_last[8]
    Sno = x_last[9]
    Snh = x_last[10]
    Snd = x_last[11]
    Xnd = x_last[12]
    Salk = x_last[13]


        # Kinetic parameter
    i_xb = 0.08
    i_xp = 0.06
    fp = 0.08


    # EQI
        # weighting factor
    B_ss = 2
    B_COD = 1
    B_NKj = 30
    B_NO = 10
    B_BOD = 2

    Snkj = Snh + Snd + Xnd + i_xb*(Xbh+Xba) + i_xp*(Xp+Xi)
    SS = 0.75*(Xs + Xi + Xbh + Xba + Xp)
    BOD5 = 0.25*(Ss + Xs + (1-fp)*(Xbh + Xba))
    COD = Ss + Si + Xs + Xi + Xbh + Xba + Xp

    EQI = (B_ss*SS + B_COD*COD+B_NKj*Snkj+ B_NO*Sno+B_BOD*BOD5)*(1/1000)*0.66    #Eff는 draw phase 농도에서 계산해야함.. 디테일수정 필요, Qeff도, 그리고 단위 맞출 필요가 있음.

    """
    Basic phase sequencing:

            Phase No./      Feeding     Aeration    Mixing      Discharge/  Type
            length(%)                                           Wastage
            1 (4.2)         Yes         No          Yes         No          FLL/Rxn (ANX)    
            2 (8.3)         No          No          Yes         No          Rxn (ANX)
            3 (37.5)        No          Yes         Yes         No          Rxn (AER)
            4 (31.2)        No          No          Yes         No          Rxn (ANX)
            5 (2.1)         No          Yes         Yes         No          Rxn (AER)
            6 (8.3)         No          No          No          No          STL
            7 (2.1)         No          No          No          Yes         DRW
            8 (6.3)         No          Yes         No          No          IDL

            (ref. Pons et al. Tbl. 1) 
    """

    # OCI
    dt =  DO_control_par[2]
    So_sat = DO_control_par[10]
    T = 0.5 # 12hrs, 0.5 day
        #Mechanical Eergy (kWh / d)
    t = 12 #
    ME_1 = 0.005*1.32*0.021
    ME_2 = 0.005*1.32*0.0415
    ME_3 = 0.005*1.32*0.1875
    ME_4 = 0.005*1.32*0.156
    ME_5 = 0.005*1.32*0.0105

    ME = (24/T)*(ME_1 + ME_2 + ME_3 + ME_4 + ME_5)

    # Aeration energy (kWh / d)
    AE_3 = 1.32* sum(kla_memory_3)*dt
    AE_5 = 1.32* sum(kla_memory_5)*dt
    AE_8 = 1.32* sum(kla_memory_8)*dt

    AE = So_sat/(T*1.8*1000)*(AE_3+ AE_5 +AE_8)

    # Pumping energy (kWh / d)

    PE = (1/T)*( 0.05* Qw) # SBR에는 내부 외부 반송 없음

    OCI = AE + PE + ME #+ 5*SP + 3*EC

    reward = -(EQI + 0.0005* OCI)


    return reward





