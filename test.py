import json

FILENAME_TRAIN = '/DATA/xuehy/logs/maskable_ppo_seed0_20220909103309/370000_steps_cache.json'
FILENAME_VALIDATION = '/DATA/xuehy/preload/zz300_dynamic_20190101_20211231.json'

if __name__ == '__main__':
    with open(FILENAME_TRAIN) as f:
        train = json.load(f)['valid']
    with open(FILENAME_VALIDATION) as f:
        valid = json.load(f)['valid']

    threshold = 0.05
    train_selected = set(key for key in train if train[key] >= threshold)
    valid_selected = set(key for key in valid if valid[key] >= threshold)

    for key in train_selected & valid_selected:
        print(key, train[key], valid[key])
