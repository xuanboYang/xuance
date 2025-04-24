import os
import copy
import numpy as np
from xuance.oneflow.runners import RunnerBase
from xuance.oneflow.agents import REGISTRY_Agents
from xuance.environment import make_envs


class RunnerMARL(RunnerBase):
    def __init__(self, config):
        super(RunnerMARL, self).__init__(config)
        self.agents = REGISTRY_Agents[config.agent](config, self.envs)
        self.config = config

        if self.agents.distributed_training:
            self.rank = int(os.environ['RANK'])

    def run(self):
        if self.config.test_mode:
            def env_fn():
                config_test = copy.deepcopy(self.config)
                config_test.parallels = 1
                config_test.render = True
                return make_envs(config_test)
            self.agents.render = True
            
            # 检查模型目录是否存在，如果不存在则创建
            model_dir = self.agents.model_dir_load
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                print(f"创建模型目录: {model_dir}")
            
            # 尝试加载模型，如果失败则进行简单训练以创建模型
            try:
                self.agents.load_model(self.agents.model_dir_load)
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("将进行简单训练以创建模型...")
                # 进行少量训练步骤以创建模型
                n_train_steps = 10  # 只训练少量步骤
                self.agents.train(n_train_steps)
                # 保存训练后的模型
                self.agents.save_model("test_model.pth")
                # 重新加载模型
                self.agents.load_model(self.agents.model_dir_load)
            
            scores = self.agents.test(env_fn, self.config.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            print("开始训练模式，步数:", self.config.running_steps)
            n_train_steps = self.config.running_steps // self.n_envs
            print(f"计算训练步数: {n_train_steps} = {self.config.running_steps} // {self.n_envs}")
            try:
                print("开始训练...")
                self.agents.train(n_train_steps)
                print("训练完成.")
                self.agents.save_model("final_train_model.pth")
                print("模型保存完成.")
            except Exception as e:
                print(f"训练过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

        self.agents.finish()
        self.envs.close()

    def benchmark(self):
        def env_fn():
            config_test = copy.deepcopy(self.config)
            config_test.parallels = 1  # config_test.test_episode
            return make_envs(config_test)

        train_steps = self.config.running_steps // self.n_envs
        eval_interval = self.config.eval_interval // self.n_envs
        test_episode = self.config.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = self.agents.test(env_fn, test_episode) if self.rank == 0 else 0.0
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agents.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agents.train(eval_interval)
            if self.rank == 0:
                test_scores = self.agents.test(env_fn, test_episode)

                if np.mean(test_scores) > best_scores_info["mean"]:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": self.agents.current_step}
                    # save best model
                    self.agents.save_model(model_name="best_model.pth")

        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
        self.agents.finish()
        self.envs.close()
