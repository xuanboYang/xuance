# -*- coding: utf-8 -*-

from argparse import Namespace
from xuance import get_runner
import unittest

n_steps = 10000

device = 'npu'
test_mode = False  # False: 训练模型  True: 加载模型


class MyTestCase(unittest.TestCase):
    """
    多智能体支持的算法包括（IQL VDN QMIX MADDPG QTRAN COMA  MAPPO）
    """

    def test_iql(self):
        args = Namespace(**dict(dl_toolbox='paddle', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="iql", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()
        # 显式使用 self，避免静态方法提示
        if self:
            assert True
    
    def test_vdn(self):
        args = Namespace(**dict(dl_toolbox='paddle', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="vdn", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()
        # 显式使用 self，避免静态方法提示 
        if self:
            assert True
    
    def test_qmix(self):
        args = Namespace(**dict(dl_toolbox='paddle', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="qmix", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()
        # 显式使用 self，避免静态方法提示
        if self:
            assert True
   
    def test_maddpg(self):
        args = Namespace(**dict(dl_toolbox='paddle', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="maddpg", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()
        # 显式使用 self，避免静态方法提示
        if self:
            assert True
    
    def test_qtran(self):
        args = Namespace(**dict(dl_toolbox='paddle', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="qtran", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()
        # 显式使用 self，避免静态方法提示
        if self:
            assert True
    
    def test_coma(self):
        args = Namespace(**dict(dl_toolbox='paddle', device=device, running_steps=n_steps, test_mode=test_mode, buffer_size=1000))
        runner = get_runner(method="coma", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()
        # 显式使用 self，避免静态方法提示
        if self:
            assert True
    
    def test_mappo(self):
        args = Namespace(**dict(dl_toolbox='paddle', device=device, running_steps=n_steps, test_mode=test_mode, buffer_size=1000))
        runner = get_runner(method="mappo", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()
        # 显式使用 self，避免静态方法提示
        if self:
            assert True


if __name__ == '__main__':
    unittest.main()
