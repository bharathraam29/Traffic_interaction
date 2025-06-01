# Predictive Decision-making with Sparse Autoencoder

This repository is a modified version of the original [Learning Interaction-aware Motion Prediction Model](https://github.com/MCZhi/Predictive-Decision) project, implementing a Sparse Autoencoder (SAE) approach instead of the original LSTM/GRU architecture.

## Key Modifications

- Replaced LSTM/GRU with Sparse Autoencoder in the trajectory prediction model
- Added sparsity constraints for better feature extraction
- Maintained the original interaction-aware framework

## Architecture Details

The Sparse Autoencoder implementation includes:
- Encoder: Input → Hidden (384) → Latent (128)
- Decoder: Latent (128) → Hidden (384) → Output (3)
- Sparsity target: 0.05 (5% target activation)
- KL divergence-based sparsity penalty

## Installation

### Create a new Conda environment
```bash
conda create -n smarts python=3.8
```

### Install the SMARTS simulator
```bash
conda activate smarts

# Download SMARTS
git clone https://github.com/huawei-noah/SMARTS.git
cd <path/to/SMARTS>
git checkout comp-1

# Install the system requirements
bash utils/setup/install_deps.sh

# Install smarts with comp-1 branch
pip install "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1"
```

### Install PyTorch
```bash
conda install pytorch==1.12.0 -c pytorch
```

## Training

Run `train.py` with the following arguments:
```bash
python train.py --name SAE --use_exploration --use_interaction
```

## Testing

Run `test.py` with your trained model:
```bash
python test.py --model_path /training_log/SAE/model.pth
```

To visualize in Envision:
```bash
scl envision start -p 8081
```
Then go to `http://localhost:8081/`

## Citation

If you use this code in your research, please cite both the original paper and this implementation:

```bibtex
@article{huang2023learning,
  title={Learning Interaction-aware Motion Prediction Model for Decision-making in Autonomous Driving},
  author={Huang, Zhiyu and Liu, Haochen and Wu, Jingda and Huang, Wenhui and Lv, Chen},
  journal={arXiv preprint arXiv:2302.03939},
  year={2023}
}
```
