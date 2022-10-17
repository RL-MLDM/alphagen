from matplotlib import pyplot as plt

from tensorflow.train import summary_iterator

MODEL_NAME = 'ppo_e_lstm'
TB_LOG_DIR = f'/DATA/xuehy/tb_logs/{MODEL_NAME}'
EVENT_PREFIX = 'ppo_e_lstm_s0_p20_20221009122614_1/events.out.tfevents.1665289575.cloud88.50175.0'
PATH = f'{TB_LOG_DIR}/{EVENT_PREFIX}'

if __name__ == '__main__':
    test_ic = []
    valid_ic = []
    train_ic = []

    for record in summary_iterator(PATH):
        for value in record.summary.value:
            if value.tag == 'test/ic_':
                test_ic.append(value.simple_value)
            elif value.tag == 'valid/ic':
                valid_ic.append(value.simple_value)
            elif value.tag == 'pool/best_ic_ret':
                train_ic.append(value.simple_value)

    print(len(train_ic), len(valid_ic), len(test_ic))
    timestamps = [i * 2048 for i in range(1, len(train_ic) + 1)]

    plt.plot(timestamps, train_ic, label='train')
    plt.plot(timestamps, valid_ic, label='valid')
    plt.plot(timestamps, test_ic, label='test')

    plt.legend()
    plt.show()
