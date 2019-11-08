'''
'''
'''
res18 网络配置参数
'''
config_res18 = {}
config_res18['anchors'] = [10.0, 30.0, 60.]
config_res18['chanel'] = 1
config_res18['crop_size'] = [128, 128, 128]
config_res18['stride'] = 4
config_res18['max_stride'] = 16
config_res18['num_neg'] = 800
config_res18['th_neg'] = 0.02
config_res18['th_pos_train'] = 0.5
config_res18['th_pos_val'] = 1
config_res18['num_hard'] = 2
config_res18['bound_size'] = 12
config_res18['reso'] = 1
config_res18['sizelim'] = 6.  # mm
config_res18['sizelim2'] = 30
config_res18['sizelim3'] = 40
config_res18['aug_scale'] = True
config_res18['r_rand_crop'] = 0.3
config_res18['pad_value'] = 170
config_res18['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config_res18['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']
config_res18['margin'] = 32
config_res18['sidelen'] = 144
