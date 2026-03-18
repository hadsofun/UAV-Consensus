import os, sys, datetime, platform

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from observer.RobustDifferentatior_3rd import robust_differentiator_3rd as rd3
from uav.uav_consensus import uav_consensus, uav_param
from utils.ref_cmd import *
from utils.utils import *
from consensus_uncertainty import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cur_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
cur_path = os.path.dirname(os.path.abspath(__file__))
windows = platform.system().lower() == 'windows'
print(cur_path)
if windows:
    new_path = cur_path + '\\..\\..\\datasave\\pos_consensus-' + cur_time + '/'
else:
    new_path = cur_path + '/../../datasave/pos_consensus-' + cur_time + '/'

config_file = cur_path + '/global_configuration.xml'
config_root = XML_Load(config_file)  # 加载全局配置文件
g_v = get_global_variable_from_XML(config_root)  # 加载一些全局变量
uav_par = get_uav_param_from_XML(config_root)
uav_par['dt'] = g_v['dt']  # 仅仅是为了防止采样周期不一样，全部以global variable为准
uav_par['g_tm'] = g_v['g_tm']  # 仅仅是为了防止采样周期不一样，全部以global variable为准
uav_par = uav_param(from_dict=uav_par)
# print(uav_par.time_max)
att_ctrl_param = get_att_ctrl_param_from_XML(config_root)
att_ctrl_param.dt = g_v['dt']

pos_ctrl_param = get_pos_ftpd_ctrl_param_from_XML(config_root)
pos_ctrl_param.dt = g_v['dt']

TEST_GROUP = 0
# 0: 大圈逆时针，小圈不动
# 1: 整体平移，小圈不动
# 2: 第三组 大圈逆时针，小圈逆时针
# 3: 大圈8字，小圈不动
# else: 不动

'''uav group initialization'''
USE_RL = False
USE_OBS_IN = False
USE_OBS_OUT = True
SAVE = True

'''generate uncertainty and global reference for all UAVs at all timesteps'''
consensus_un = random_uncertainty_n(g_v['uav_num'], g_v['dt'], g_v['g_tm'], g_v['g_ideal'])
REF, DOT_REF, DOT2_REF, NU, DOT_NU, DOT2_NU, pos0 = ref_uav_consensus_sequence(g_v['dt'], g_v['g_tm'], TEST_GROUP)

uavs = []
for i in range(g_v['uav_num']):
    obs_in = rd3(use_freq=True, omega=np.array([3.5, 3.4, 4]), dim=3, thresh=np.array([0.5, 0.5, 0.5]), dt=g_v['dt'])
    obs_in.set_init(all_zero=True)
    obs_out = rd3(use_freq=True, omega=np.array([3.5, 3.5, 2.8]), dim=3, thresh=np.array([0.5, 0.5, 0.5]), dt=g_v['dt'])
    obs_out.set_init(all_zero=True)
    uav_con = usv_consensus(uav_param=uav_par,
                            ctrl_att_param=att_ctrl_param,
                            ctrl_pos_param=pos_ctrl_param,
                            adjacency=g_v['g_A'][i],
                            in_degree=g_v['g_D'][i],
                            communication=g_v['g_B'][i],
                            obs_att=obs_in,
                            obs_pos=obs_out,
                            is_ideal=g_v['g_ideal'])
    uav_con.uav.set_pos0(pos0[i])
    base_path = uav_con.uav.project_path + 'comparative_cost/position_single/'
    uavs.append(uav_con)
'''uav group initialization'''


def cal_g_eta_dot_eta():
    _res = np.zeros((g_v['uav_num'], 3))
    _dot_res = np.zeros((g_v['uav_num'], 3))
    for p in range(g_v['uav_num']):
        _res[p][:] = uavs[p].uav.eta()
        _dot_res[p][:] = uavs[p].uav.dot_eta()
    return _res, _dot_res


def cal_g_obs_2nd_dynamic(g_dd_nu: np.ndarray):
    _g_obs = np.zeros((g_v['uav_num'], 3))
    _dy = np.zeros((g_v['uav_num'], 3))
    for p in range(g_v['uav_num']):
        _g_obs[p][:] = uavs[p].obs_pos.z3
        _dy[p][:] = -uavs[p].uav.kt / uavs[p].uav.m * uavs[p].uav.dot_eta() + uavs[p].ctrl_pos.control_out_consensus - g_dd_nu[p]
    return _g_obs, _dy


if __name__ == '__main__':
    # '''初始化一下，没啥用，放着'''
    # for i in range(g_v['uav_num']):
    #     uavs[i].reset(uav_par, att_ctrl_param, pos_ctrl_param)
    # '''初始化一下，没啥用，放着'''

    while g_v['g_t'] < g_v['g_tm'] - g_v['dt'] / 2:
        '''嗨嗨嗨'''
        # 刚开始的时候，可能e比较大，所以如果采用双向拓扑，那么会有超调
        # 所以，刚开始几秒中不使用双向拓扑，等到基本没误差，再转化为双向的，美滋滋
        # 实际是否使用，可以酌情调试
        # if g_v['g_t'] > 5.0:
        #     uavs[0].adjacency = np.array([0, 1, 1, 1])
        #     uavs[0].d = 3
        '''嗨嗨嗨'''

        if g_v['g_N'] % int(1 / g_v['dt']) == 0:
            print('time: %.2f s.' % (g_v['g_N'] / int(1 / g_v['dt'])))

        '''2. calculations for each UAV'''
        ref, dot_ref, dot2_ref = REF[g_v['g_N']], DOT_REF[g_v['g_N']], DOT2_REF[g_v['g_N']]
        # ref[2] = ref_bias_a[2] + 0.5 * g_v['g_t']
        # dot_ref[2] = 0.5

        nu, dot_nu, dot2_nu = NU[g_v['g_N']], DOT_NU[g_v['g_N']], DOT2_NU[g_v['g_N']]

        g_eta, g_dot_eta = cal_g_eta_dot_eta()  # 先计算全局状态
        g_obs, g_2nd_dynamics = cal_g_obs_2nd_dynamic(dot2_nu)

        for i in range(g_v['uav_num']):  # 对于每一个无人机
            '''2.1 generate reference command, uncertainty, and bias for each uav'''
            dis_i = consensus_un[g_v['g_N'], 6 * i: 6 * (i + 1)]

            eta_d_i = ref[0: 3]  # eta 表示外环，d表示参考，i表示无人机编号
            dot_eta_d_i = dot_ref[0: 3]  # dot 表示一阶导数，eta表示外环，d表示参考，i表示无人机编号
            dotdot_eta_d_i = dot2_ref[0: 3]

            if USE_OBS_OUT:
                syst_dynamic_i = -uavs[i].uav.kt / uavs[i].uav.m * uavs[i].uav.dot_eta() + uavs[i].uav.A()
                obs_eta_i, _ = uavs[i].obs_pos.observe(x=uavs[i].uav.eta(), syst_dynamic=syst_dynamic_i)
            else:
                obs_eta_i = np.zeros(3)

            '''2.2 consensus control update'''
            uavs[i].cal_consensus_e(g_eta, nu, nu[i], eta_d_i, sat=True, thresh=np.array([2.5, 2.5, 2.8]))  # 计算 e
            uavs[i].cal_consensus_dot_e(g_dot_eta, dot_nu, dot_nu[i], dot_eta_d_i)  # 计算 de
            Lambda_eta = uavs[i].cal_Lambda_eta(g_2nd_dynamics, dot2_nu[i], dotdot_eta_d_i)

            uavs[i].calculate_cost(eta_d_i, nu[i], dot_eta_d_i, dot_nu[i])

            uavs[i].ctrl_pos.control_update_outer_consensus(d=uavs[i].d,
                                                            b=uavs[i].b,
                                                            consensus_e=uavs[i].consensus_e,
                                                            consensus_de=uavs[i].consensus_dot_e,
                                                            dd_ref=dot2_ref[0:3],
                                                            dd_nu=dot2_nu[i])

            '''2.3 transfer virtual control command to actual throttle, phi_d, and theta_d'''
            phi_d_old = uavs[i].rho_d[0]
            theta_d_old = uavs[i].rho_d[1]
            phi_d, theta_d, dot_phi_d, dot_theta_d, throttle = uo_2_ref_angle_throttle(uo=uavs[i].ctrl_pos.control_out_consensus,
                                                                                       att=uavs[i].uav.uav_att(),
                                                                                       m=uavs[i].uav.m,
                                                                                       g=uavs[i].uav.g,
                                                                                       phi_d_old=phi_d_old,
                                                                                       theta_d_old=theta_d_old,
                                                                                       dt=uavs[i].uav.dt,
                                                                                       att_limit=[np.pi / 3, np.pi / 3], #
                                                                                       dot_att_limit=[np.pi / 2, np.pi / 2])  # None

            uavs[i].rho_d = np.array([phi_d, theta_d, ref[3]])  # phi_d theta_d psi_d
            uavs[i].dot_rho_d = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])  # phi_d theta_d psi_d 的一阶导数
            uavs[i].throttle = throttle

            '''2.4 inner loop control'''
            e_rho_i = uavs[i].uav.rho1() - uavs[i].rho_d
            de_rho_i = np.dot(uavs[i].uav.W(), uavs[i].uav.rho2()) - uavs[i].dot_rho_d

            if USE_OBS_IN:
                syst_dynamic = (np.dot(uavs[i].uav.dW(), uavs[i].uav.omega()) +
                                np.dot(uavs[i].uav.W(), uavs[i].uav.A_omega() +
                                       np.dot(uavs[i].uav.B_omega(), uavs[i].ctrl_att.control_in)))
                obs_rho_i, _ = uavs[i].obs_att.observe(syst_dynamic=syst_dynamic, x=uavs[i].uav.rho1())
            else:
                obs_rho_i = np.zeros(3)

            # 将观测器输出前两维度置为 0 即可
            obs_rho_i[0] = 0.
            obs_rho_i[1] = 0.
            # 将观测器输出前两维度置为 0 即可

            uavs[i].ctrl_att.control_update_inner(e_rho=e_rho_i,
                                                  dot_e_rho=de_rho_i,
                                                  dd_ref=np.zeros(3),
                                                  W=uavs[i].uav.W(),
                                                  dW=uavs[i].uav.dW(),
                                                  omega=uavs[i].uav.omega(),
                                                  A_omega=uavs[i].uav.A_omega(),
                                                  B_omega=uavs[i].uav.B_omega(),
                                                  obs=obs_rho_i,
                                                  att_only=False)

            '''2.5 rk44 update'''
            action_4_uav_i = np.array([throttle, uavs[i].ctrl_att.control_in[0], uavs[i].ctrl_att.control_in[1], uavs[i].ctrl_att.control_in[2]])
            uavs[i].uav.rk44(action=action_4_uav_i, dis=dis_i, n=1, att_only=False)

            '''2.6 data record'''
            data_block_i = {'time': uavs[i].uav.time,
                            'control': action_4_uav_i,
                            'ref_angle': uavs[i].rho_d,
                            'ref_pos': ref[0: 3] + nu[i],
                            'ref_vel': dot_ref[0: 3] + dot_nu[i],
                            'd_in': np.array([0., 0., np.dot(uavs[i].uav.W(), np.array([dis_i[3], dis_i[4], dis_i[5]]))[2]]),
                            'd_in_obs': obs_rho_i,
                            'd_in_e_1st': uavs[i].obs_att.obs_error,
                            'd_out': np.array([dis_i[0], dis_i[1], dis_i[2]]) / uavs[i].uav.m,
                            'd_out_obs': obs_eta_i,
                            'd_out_e_1st': uavs[i].obs_pos.obs_error,
                            'state': uavs[i].uav.uav_state_call_back()}
            uavs[i].data_record.record(data=data_block_i)

        g_v['g_t'] += g_v['dt']  # time update
        g_v['g_N'] += 1  # global index update

    '''将每个无人机的 cost 打印出来'''
    cost = 0
    for j in range(g_v['uav_num']):
        print('uav %.0f:  %.3f' % (j, uavs[j].cost))
        cost += uavs[j].cost
    print('Total cost: %.3f' % (cost))
    '''3. draw curve'''
    DRAW = True
    if DRAW:
        data_block = []
        for _uav in uavs:
            data_block.append(_uav.data_record)
        plot_consensus_pos(data_block)
        # plot_consensus_vel(data_block)
        plot_consensus_att(data_block)
        # plot_consensus_throttle(data_block)
        # plot_consensus_torque(data_block)
        plot_consensus_outer_obs(data_block)
    plt.show()

    SAVE = True
    if SAVE:
        os.mkdir(new_path)
        for i in range(g_v['uav_num']):
            path = new_path + '\\uav_' + str(i) + '/' if windows else new_path + '/uav_' + str(i) + '/'
            os.mkdir(path)
            uavs[i].data_record.package2file(path)
