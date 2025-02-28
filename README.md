The diffrep and image-based metaworld parts are still under debugging and refinement.

## Installation

```bash
# Create environemnt
conda create -n dsac python=3.10 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate dsac

# One of: Install jax WITH CUDA 
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install package
pip install -r requirements.txt
pip install -e .
```



## Run
```bash
# Run one experiment
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py --alg sdac --seed 100
```

## Visualize results
```python
from relax.utils.inspect_results import load_results, plot_mean

env_name = 'Ant-v4'

patterns_dict = {
        'sdac': r'sdac.*' # regex expression of saved folders
    }

for key, value in patterns_dict.items():
    print(key)
    _ = load_results(value, env_name, show_df=False)

plot_mean(patterns_dict, env_name)
```

We also provide a sample log that can be visualized via directly running the commands.

```python 
from relax.utils.inspect_results import load_results, plot_mean

env_name = 'Ant-v4'

patterns_dict = {
        'sdac': r'.*sdac.*' # regex expression of saved folders
    }

# For show the sample log
from pathlib import Path
import relax
package_path = Path(relax.__file__)
logdir = package_path.parent / 'sample_logs'
for key, value in patterns_dict.items():
    print(key)
    _ = load_results(value, env_name, show_df=False, path=logdir)
plot_mean(patterns_dict, env_name, path=logdir)
```



