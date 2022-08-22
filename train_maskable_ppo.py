import os

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.utils.cache import LRUCache
from alphagen.utils.random import reseed_everything
from alphagen_qlib.evaluation import QLibEvaluation


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 name_prefix: str = "rl_model",
                 verbose: int = 0
                 ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_checkpoint()
        if self.n_calls % self.show_freq == 0:
            self.show_top_alphas()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('cache/size', len(self.cache))
        self.logger.record('cache/gt_0.03', self.cache.greater_than_count(0.03))
        self.logger.record('cache/top_1%', self.cache.quantile(0.99))
        self.logger.record('cache/top_100_avg', self.cache.top_k_average(100))

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
        self.model.save(path)
        self.cache.save(path + '_cache.json')
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")

    def show_top_alphas(self):
        if self.verbose > 0:
            top_5 = self.cache.top_k(5)
            for key in top_5:
                print(key, top_5[key])

    @property
    def cache(self) -> LRUCache:
        return self.training_env.envs[0].unwrapped.eval.cache


if __name__ == '__main__':
    reseed_everything(0)

    device = torch.device("cpu")
    csi300_2016 = ['SZ000001', 'SZ000002', 'SZ000063', 'SZ000069', 'SZ000166', 'SZ000333', 'SZ000538', 'SZ000625',
                   'SZ000651', 'SZ000725', 'SZ000776', 'SZ000858', 'SZ000895', 'SZ002024', 'SZ002252', 'SZ002304',
                   'SZ002415', 'SZ002594', 'SZ002736', 'SZ002739', 'SZ300059', 'SZ300104', 'SH600000', 'SH600010',
                   'SH600011', 'SH600015', 'SH600016', 'SH600018', 'SH600019', 'SH600023', 'SH600028', 'SH600030',
                   'SH600031', 'SH600036', 'SH600048', 'SH600050', 'SH600104', 'SH600111', 'SH600115', 'SH600150',
                   'SH600276', 'SH600340', 'SH600372', 'SH600398', 'SH600485', 'SH600518', 'SH600519', 'SH600585',
                   'SH600637', 'SH600690', 'SH600705', 'SH600795', 'SH600837', 'SH600886', 'SH600887', 'SH600893',
                   'SH600900', 'SH600958', 'SH600959', 'SH600999', 'SH601006', 'SH601018', 'SH601088', 'SH601111',
                   'SH601166', 'SH601169', 'SH601186', 'SH601211', 'SH601225', 'SH601288', 'SH601318', 'SH601328',
                   'SH601336', 'SH601377', 'SH601390', 'SH601398', 'SH601600', 'SH601601', 'SH601618', 'SH601628',
                   'SH601633', 'SH601668', 'SH601669', 'SH601688', 'SH601727', 'SH601766', 'SH601788', 'SH601800',
                   'SH601808', 'SH601818', 'SH601857', 'SH601898', 'SH601899', 'SH601901', 'SH601939', 'SH601985',
                   'SH601988', 'SH601989', 'SH601998', 'SH603288', 'SZ000009', 'SZ000027', 'SZ000039', 'SZ000046',
                   'SZ000060', 'SZ000061', 'SZ000100', 'SZ000156', 'SZ000157', 'SZ000338', 'SZ000400', 'SZ000402',
                   'SZ000413', 'SZ000415', 'SZ000423', 'SZ000425', 'SZ000503', 'SZ000539', 'SZ000540', 'SZ000559',
                   'SZ000568', 'SZ000581', 'SZ000598', 'SZ000623', 'SZ000629', 'SZ000630', 'SZ000686', 'SZ000709',
                   'SZ000712', 'SZ000728', 'SZ000729', 'SZ000738', 'SZ000750', 'SZ000768', 'SZ000778', 'SZ000783',
                   'SZ000792', 'SZ000793', 'SZ000800', 'SZ000825', 'SZ000826', 'SZ000831', 'SZ000876', 'SZ000883',
                   'SZ000898', 'SZ000917', 'SZ000937', 'SZ000963', 'SZ000983', 'SZ000999', 'SZ001979', 'SZ002007',
                   'SZ002008', 'SZ002038', 'SZ002065', 'SZ002081', 'SZ002129', 'SZ002142', 'SZ002146', 'SZ002153',
                   'SZ002195', 'SZ002202', 'SZ002230', 'SZ002236', 'SZ002241', 'SZ002292', 'SZ002294', 'SZ002353',
                   'SZ002375', 'SZ002385', 'SZ002399', 'SZ002410', 'SZ002422', 'SZ002450', 'SZ002456', 'SZ002465',
                   'SZ002470', 'SZ002475', 'SZ002500', 'SZ002673', 'SZ300002', 'SZ300003', 'SZ300015', 'SZ300017',
                   'SZ300024', 'SZ300027', 'SZ300058', 'SZ300070', 'SZ300124', 'SZ300133', 'SZ300144', 'SZ300146',
                   'SZ300251', 'SZ300315', 'SH600005', 'SH600008', 'SH600009', 'SH600021', 'SH600027', 'SH600029',
                   'SH600038', 'SH600060', 'SH600066', 'SH600068', 'SH600085', 'SH600089', 'SH600100', 'SH600109',
                   'SH600118', 'SH600153', 'SH600157', 'SH600166', 'SH600170', 'SH600177', 'SH600188', 'SH600196',
                   'SH600208', 'SH600221', 'SH600252', 'SH600256', 'SH600271', 'SH600309', 'SH600315', 'SH600317',
                   'SH600332', 'SH600350', 'SH600352', 'SH600362', 'SH600369', 'SH600373', 'SH600383', 'SH600406',
                   'SH600415', 'SH600489', 'SH600535', 'SH600547', 'SH600549', 'SH600570', 'SH600578', 'SH600583',
                   'SH600588', 'SH600600', 'SH600633', 'SH600642', 'SH600648', 'SH600649', 'SH600660', 'SH600663',
                   'SH600674', 'SH600688', 'SH600703', 'SH600717', 'SH600718', 'SH600739', 'SH600741', 'SH600783',
                   'SH600804', 'SH600820', 'SH600827', 'SH600839', 'SH600863', 'SH600867', 'SH600873', 'SH600875',
                   'SH600895', 'SH600998', 'SH601009', 'SH601016', 'SH601021', 'SH601098', 'SH601099', 'SH601106',
                   'SH601117', 'SH601118', 'SH601158', 'SH601179', 'SH601198', 'SH601216', 'SH601231', 'SH601238',
                   'SH601258', 'SH601333', 'SH601555', 'SH601607', 'SH601608', 'SH601699', 'SH601718', 'SH601866',
                   'SH601872', 'SH601888', 'SH601919', 'SH601928', 'SH601933', 'SH601958', 'SH601969', 'SH601991',
                   'SH601992', 'SH603000', 'SH603885', 'SH603993']
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    ev = QLibEvaluation(
        instrument=csi300_2016,
        start_time='2016-01-01',
        end_time='2018-12-31',
        target=target,
        device=device
    )
    env = AlphaEnv(ev)

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./logs/',
        name_prefix='maskable_ppo',
        verbose=1
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        gamma=1.,
        ent_coef=0.02,
        verbose=1
    )
    model.learn(total_timesteps=500000, callback=checkpoint_callback)
