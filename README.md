# Efficient Transformer Fine-tuning: LoRA vs QLoRA

A comprehensive comparison of parameter-efficient fine-tuning methods for LLMs, demonstrating how to achieve significant memory savings while maintaining model performance.

## Project Overview

This project compares two state-of-the-art parameter-efficient fine-tuning techniques:
- **LoRA (Low-Rank Adaptation)**: Freezes pre-trained weights and injects trainable rank decomposition matrices
- **QLoRA (Quantized LoRA)**: Combines LoRA with 4-bit quantization for extreme memory efficiency

**Model**: Facebook OPT-350M  
**Task**: Sentiment analysis on IMDB movie reviews  

## Key Results

| Metric | LoRA | QLoRA | Improvement |
|--------|------|-------|-------------|
| **Model Size** | 637.71 MB | 257.87 MB | **2.47x smaller** |
| **Trainable Parameters** | 1.57M (0.47%) | 1.57M (0.87%) | Same efficiency |
| **Training Time** | 238.64s | 357.27s | 1.5x slower |
| **Evaluation Loss** | 2.8743 | 2.9067 | Minimal difference |

### Key Insights

✅ **Memory Efficiency**: QLoRA uses **2.47x less memory** than LoRA, enabling larger models on limited hardware  
✅ **Parameter Efficiency**: Both methods fine-tune only **0.47-0.87%** of model parameters  
✅ **Performance**: Negligible performance difference (Δ loss: 0.0323)  
⚠️ **Trade-off**: QLoRA is 1.5x slower due to quantization overhead

## Quick Start

```bash
# Clone and open in Google Colab
# Install dependencies (included in notebook)
!pip install transformers datasets peft bitsandbytes accelerate

# Run the notebook end-to-end
```
