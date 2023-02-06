# AlphaGen

Automatic formulaic alpha generation with reinforcement learning.

## How to reproduce?

### Data preparation

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


