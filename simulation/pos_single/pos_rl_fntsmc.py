import os, sys, datetime, platform, torch
import matplotlib.pyplot as plt

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from observer.RobustDifferentatior_3rd import robust_differentiator_3rd as rd3
from controller.FNTSMC import fntsmc, fntsmc_param
from uav.uav import UAV, uav_param
from utils.ref_cmd import *
from utils.utils import *
from utils.collector import data_collector
from utils.PPOActor import PPOActor_Gaussian

'''Parameter list of the quadrotor'''
DT = 0.001
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 60
'''Parameter list of the quadrotor'''

'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param(
    # k1=np.array([6.00810648, 6.80311651, 13.47563418]).astype(float),  # 手调: 4 4 15
    # k2=np.array([2.04587905, 1.60844957, 0.98401018]).astype(float),  # 手调: 1 1 1.5
    # k3=np.array([0.05, 0.05, 0.05]).astype(float),
    # k4=np.array([9.85776965, 10.91725924, 13.90115023]).astype(float),  # 要大     手调: 5 4 5
    k1=np.array([4., 4., 15.]).astype(float),
    k2=np.array([1., 1., 1.5]).astype(float),
    k3=np.array([0.05, 0.05, 0.05]).astype(float),
    k4=np.array([5, 4, 5]).astype(float),  # 要大
    alpha1=np.array([1.01, 1.01, 1.01]).astype(float),
    alpha2=np.array([1.01, 1.01, 1.01]).astype(float),
    dim=3,
    dt=DT
    
    # k1 控制滑模中 e 的占比，k3 控制滑模中 sig(e)^alpha1 的占比
    # k2-alpha1 的组合不要太大
    # k3 是观测器补偿用的，实际上观测的都很好，所以 k4 要大于0，但是非常小
    # k4-alpha2 的组合要比k3-alpha1 大，但是也别太大
)
'''Parameter list of the attitude controller'''

'''Parameter list of the position controller'''
pos_ctrl_param = fntsmc_param(
    k1=np.array([0.3, 0.3, 1.0]),
    k2=np.array([0.5, 0.5, 1]),
    k3=np.array([3, 3, 3]),  # 补偿观测器的，小点就行
    k4=np.array([6, 6, 6]),
    alpha1=np.array([1.01, 1.01, 1.01]),
    alpha2=np.array([1.01, 1.01, 1.01]),
    dim=3,
    dt=DT
)
'''Parameter list of the position controller'''

IS_IDEAL = False
USE_OBS = True
USE_RL = True

cur_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
cur_path = os.path.dirname(os.path.abspath(__file__))
windows = platform.system().lower() == 'windows'
if windows:
    new_path = cur_path + '\\..\\..\\datasave\\pos_bs_fntsmc_bs_fntsmc-' + cur_time + '/'
else:
    new_path = cur_path + '/../../datasave/pos_bs_fntsmc_bs_fntsmc-' + cur_time + '/'

if __name__ == '__main__':
    uav = UAV(uav_param)
    ctrl_in = fntsmc(att_ctrl_param)
    ctrl_out = fntsmc(pos_ctrl_param)
    obs_in = rd3(use_freq=True, omega=np.array([3.5, 3.4, 10]), dim=3, thresh=np.array([0.5, 0.5, 0.5]), dt=uav.dt)
    obs_out = rd3(use_freq=True, omega=np.array([3.2, 3.2, 3.2]), dim=3, thresh=np.array([0.5, 0.5, 0.5]), dt=uav.dt)
    
    actor_path = uav.project_path + 'neural_network/pos_maybe_good_2/'  # pos_maybe_good_3 画图最好，但是不适合仿真
    actor = PPOActor_Gaussian(state_dim=6, action_dim=9)
    actor.load_state_dict(torch.load(actor_path + 'actor'))
    
    uav.load_pos_normalizer(actor_path + 'state_norm.csv')
    base_path = uav.project_path + 'comparative_cost/position_single/'
    
    '''reference signal initialization'''
    ref_amplitude = np.array([2, 2, 1, deg2rad(90)])  # x y z psi
    # ref_amplitude = np.array([0, 0, 0, 0])  # x y z psi
    ref_period = np.array([5, 5, 5, 5])
    ref_bias_a = np.array([2, 2, 1, 0])
    ref_bias_phase = np.array([deg2rad(90), 0, 0, 0])
    
    '''data storage initialization'''
    data_record = data_collector(N=int(uav.time_max / uav.dt))
    
    '''reference signal'''
    phi_d = theta_d = phi_d_old = theta_d_old = 0.
    dot_phi_d = (phi_d - phi_d_old) / uav.dt
    dot_theta_d = (theta_d - theta_d_old) / uav.dt
    throttle = uav.m * uav.g
    
    '''control'''
    while uav.time < uav.time_max - uav.dt / 2:
        if uav.n % int(1 / uav.dt) == 0:
            print('time: %.2f s.' % (uav.n / int(1 / uav.dt)))
        
        '''1. generate reference command and uncertainty'''
        if uav.time < uav.time_max / 3:
            ref_amplitude = np.array([2, 2, 1, deg2rad(90)])  # x y z psi
            ref_period = np.array([5, 5, 5, 5])
            ref_bias_a = np.array([2, 2, 1, 0])
            ref_bias_phase = np.array([deg2rad(90), 0, 0, 0])
        elif uav.time_max / 3 <= uav.time < 2 * uav.time_max / 3:
            ref_amplitude = np.array([0, 0, 0, deg2rad(90)])
            ref_period = np.array([8, 8, 8, 10])
            ref_bias_a = np.array([-4, -4, 0, 0])
            ref_bias_phase = np.array([0, deg2rad(90), 0, 0])
        else:
            ref_amplitude = np.array([5, 5, 2, deg2rad(90)])
            ref_period = np.array([8, 8, 8, 10])
            ref_bias_a = np.array([5, 5, -4, 0])
            ref_bias_phase = np.array([0, deg2rad(90), 0, 0])
        ref, dot_ref, dotdot_ref = ref_uav(uav.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
        uncertainty = generate_uncertainty(time=uav.time, is_ideal=IS_IDEAL)
        
        '''2. generate outer-loop reference signal 'eta_d' and its 1st derivatives'''
        eta_d = ref[0: 3]
        e_eta = uav.eta() - eta_d
        de_eta = uav.dot_eta() - dot_ref[0: 3]
        
        '''3. generate outer-loop virtual control command'''
        if USE_OBS:
            syst_dynamic = -uav.kt / uav.m * uav.dot_eta() + uav.A()
            obs_eta, _ = obs_out.observe(x=uav.eta(), syst_dynamic=syst_dynamic)
        else:
            obs_eta = np.zeros(3)
        # obs_eta = np.zeros(3)
        '''4. outer loop control'''
        if USE_RL:
            _s = np.concatenate((e_eta, de_eta))
            new_pos_ctrl_parma = actor.evaluate(uav.pos_state_norm(_s, update=False))
            hehe = np.array([1, 1, 1, 1, 1, 1, 5, 5, 5]).astype(float)
            ctrl_out.get_param_from_actor(new_pos_ctrl_parma * hehe)
        ctrl_out.control_update_outer(e_eta=e_eta,
                                      dot_e_eta=de_eta,
                                      dot_eta=uav.dot_eta(),
                                      kt=uav.kt,
                                      m=uav.m,
                                      dd_ref=dotdot_ref[0:3],
                                      obs=obs_eta)
        
        '''5. transfer virtual control command to actual throttle, phi_d, and theta_d'''
        phi_d_old = phi_d
        theta_d_old = theta_d
        phi_d, theta_d, dot_phi_d, dot_theta_d, throttle = uo_2_ref_angle_throttle(uo=ctrl_out.control_out,
                                                           att=uav.uav_att(),
                                                           m=uav.m,
                                                           g=uav.g,
                                                           phi_d_old=phi_d_old,
                                                           theta_d_old=theta_d_old,
                                                           dt=uav.dt,
                                                           att_limit=[np.pi / 3, np.pi / 3],
                                                           dot_att_limit=[np.pi / 2, np.pi / 2])
        
        uav.sum_reward += uav.get_reward_pos(ref[0: 3], dot_ref[0:3], ctrl_out.control_out)
        
        # dot_phi_d = (phi_d - phi_d_old) / uav.dt
        # dot_theta_d = (theta_d - theta_d_old) / uav.dt
        rhod = np.array([phi_d, theta_d, ref[3]])  # phi_d theta_d psi_d
        dot_rhod = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])  # phi_d theta_d psi_d 的一阶导数
        
        '''6. inner loop control'''
        e_rho = uav.rho1() - rhod
        de_rho = np.dot(uav.W(), uav.rho2()) - dot_rhod
        
        if USE_OBS:
            syst_dynamic = np.dot(uav.dW(), uav.omega()) + np.dot(uav.W(), uav.A_omega() + np.dot(uav.B_omega(), ctrl_in.control_in))
            obs_rho, _ = obs_in.observe(syst_dynamic=syst_dynamic, x=uav.rho1())
        else:
            obs_rho = np.zeros(3)
        
        # 将观测器输出前两维度置为 0 即可
        obs_rho[0] = 0.
        obs_rho[1] = 0.
        # 将观测器输出前两维度置为 0 即可
        
        ctrl_in.control_update_inner(e_rho=e_rho,
                                     dot_e_rho=de_rho,
                                     dd_ref=np.zeros(3),
                                     W=uav.W(),
                                     dW=uav.dW(),
                                     omega=uav.omega(),
                                     A_omega=uav.A_omega(),
                                     B_omega=uav.B_omega(),
                                     obs=obs_rho,
                                     att_only=False)
        
        '''7. rk44 update'''
        action_4_uav = np.array([throttle, ctrl_in.control_in[0], ctrl_in.control_in[1], ctrl_in.control_in[2]])
        uav.rk44(action=action_4_uav, dis=uncertainty, n=1, att_only=False)
        
        '''8. data record'''
        if IS_IDEAL:
            in_obs_error = np.zeros(3)
            out_obs_error = np.zeros(3)
        else:
            in_obs_error = obs_in.obs_error
            out_obs_error = obs_out.obs_error
        data_block = {'time': uav.time,
                      'control': action_4_uav,
                      'ref_angle': rhod,
                      'ref_pos': ref[0: 3],
                      'ref_vel': dot_ref[0: 3],
                      'd_in': np.array([0., 0., np.dot(uav.W(), np.array([uncertainty[3], uncertainty[4], uncertainty[5]]))[2]]),
                      'd_in_obs': obs_rho,
                      'd_in_e_1st': in_obs_error,
                      'd_out': np.array([uncertainty[0], uncertainty[1], uncertainty[2]]) / uav.m,
                      'd_out_obs': obs_eta,
                      'd_out_e_1st': out_obs_error,
                      'state': uav.uav_state_call_back()}
        data_record.record(data=data_block)
    
    # print('reward', uav.sum_reward)
    SAVE = False
    if SAVE:
        os.mkdir(new_path)
        data_record.package2file(new_path)
    
    data_record.plot_att()
    # data_record.plot_vel()
    data_record.plot_pos()
    # data_record.plot_throttle()
    # data_record.plot_torque()
    # data_record.plot_outer_obs()
    # data_record.plot_inner_obs()
    plt.show()
