# ISATSleepNet

## Abstract 
Automatic sleep staging is critical for sleep quality assessment but hindered by dataset class imbalance, poor multimodal physiological modeling, and insufficient capture of cross-timescale sleep dynamics. We propose ISATSleepNet, a neuro-inspired model with interactive temporal attention, to address these issues: (1) Physio-GAN alleviates imbalance by generating physiologically consistent synthetic samples with temporal/spectral constraints; (2) a neuro-inspired multi-scale EEG extractor and ocular dynamics module enable efficient multimodal feature extraction; (3) a cross-modal interactive temporal attention module unifies modality interaction and temporal dependency learning. Experiments on Sleep-EDF-20 (89.1%), Sleep-EDF-78 (87.1%), and SHHS (89.7%) datasets show our method outperforms state-of-the-art approaches, validating its effectiveness and robustness.

![]([图片地址](https://github.com/JiangZaiUp/ISATSleepNet/blob/main/images/Architecture%20of%20the%20proposed%20model..png))
