# AlphaGen

Automatic formulaic alpha generation with reinforcement learning.

**Note**: There is still some documentation that needs to be done on this (and only this) page, please wait several days for the complete instructions.

## How to reproduce?

### Data preparation

- We need some of the metadata (but not the actual stock price/volume data) given by Qlib, so follow the data preparing process in [Qlib](https://github.com/microsoft/qlib#data-preparation) first.
- The actual stock data we use are retrieved from [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5), due to concerns on the timeliness and truthfulness of the data source used by Qlib.
- The data can be downloaded by running the script `data_collection/fetch_baostock_data.py`. The newly downloaded data is saved into `~/.qlib/qlib_data/cn_data_baostock_fwdadj` by default. This path can be customized to fit your specific needs, but make sure to use the correct path when loading the data (In `alphagen_qlib/stock_data.py`, function `StockData._init_qlib`, the path should be passed to qlib with `qlib.init(provider_uri=path)`).

### Before running

All principle components of our expriment are located in [train_maskable_ppo.py](train_maskable_ppo.py). You should focus on the following parameters:

- instruments (Set of instruments)
- pool_capacity (Size of combination model)
- steps (Limit of RL steps)
- batch_size (PPO batch size)
- features_extractor_kwargs (Arguments for LSTM shared net)
- seed (Random seed)
- device (PyTorch device)
- start_time & end_time (Data range for each dataset)
- save_path (Path for checkpoints)
- tensorboard_log (Path for TensorBoard)

### Run!

Simply run [train_maskable_ppo.py](train_maskable_ppo.py), or DIY if you understand our code well.

### After running


