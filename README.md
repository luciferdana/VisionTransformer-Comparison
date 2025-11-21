# Vision Transformer Comparison on CIFAR-100

Comparison of three Vision Transformer models: ViT, Swin Transformer, and DeiT on CIFAR-100 dataset.

## Models Compared

1. **Vision Transformer (ViT)** - Pure transformer architecture for vision
2. **Swin Transformer** - Hierarchical transformer with shifted window attention
3. **DeiT** - Data-efficient Image Transformer with knowledge distillation

## Dataset

- **Dataset**: CIFAR-100
- **Classes**: 100
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Image size**: 224x224 (resized from 32x32)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running on Google Colab

1. Upload the notebook to Google Colab
2. Run all cells sequentially
3. The notebook will automatically download CIFAR-100 dataset
4. Training will use GPU if available

### Running Locally

```bash
jupyter notebook vision_transformer_comparison.ipynb
```

## Metrics Evaluated

### Model Parameters
- Total parameters
- Trainable parameters
- Non-trainable parameters
- Model size (MB)

### Performance Metrics
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)
- Confusion Matrix
- Training/Validation Loss curves
- Training/Validation Accuracy curves

### Inference Time
- Average inference time per image (ms)
- Standard deviation
- Throughput (images/second)

## Training Configuration

- **Epochs**: 10
- **Batch size**: 64
- **Optimizer**: AdamW
- **Learning rate**: 1e-4
- **Weight decay**: 0.01
- **Scheduler**: CosineAnnealingLR
- **Loss function**: CrossEntropyLoss

## Output Files

The notebook generates the following files:

1. `vit_cifar100.pth` - ViT model weights
2. `swin_cifar100.pth` - Swin Transformer model weights
3. `deit_cifar100.pth` - DeiT model weights
4. `model_comparison_summary.csv` - Summary table of all metrics
5. `experiment_results.pkl` - Complete results in pickle format
6. `training_history_comparison.png` - Training curves comparison
7. `metrics_comparison.png` - Performance metrics comparison
8. `parameter_comparison.png` - Model parameters comparison
9. `inference_time_comparison.png` - Inference time comparison
10. `confusion_matrix_*.png` - Confusion matrix for each model

## Project Structure

```
.
├── vision_transformer_comparison.ipynb
├── requirements.txt
├── README.md
├── data/
│   └── cifar-100-python/
├── *.pth (model weights)
├── *.png (visualizations)
├── *.csv (results summary)
└── experiment_results.pkl
```

## Hardware Requirements

- **GPU**: Recommended (Google Colab T4 is sufficient)
- **RAM**: Minimum 8GB
- **Storage**: ~3GB for dataset and model weights

## References

1. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
3. Touvron, H., et al. (2021). "Training data-efficient image transformers & distillation through attention." ICML 2021.

## Notes

- All models use pre-trained weights from ImageNet
- Transfer learning approach with fine-tuning on CIFAR-100
- Fair comparison ensured by using same training configuration for all models
- GPU synchronization applied for accurate inference time measurement
