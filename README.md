# ISATSleepNet
### ISATSleepNet: A Neuro-Inspired Sleep Staging Network with Interactive Temporal Attention

## Abstract 
Automatic sleep staging is critical for sleep quality assessment but hindered by dataset class imbalance, poor multimodal physiological modeling, and insufficient capture of cross-timescale sleep dynamics. We propose ISATSleepNet, a neuro-inspired model with interactive temporal attention, to address these issues: (1) Physio-GAN alleviates imbalance by generating physiologically consistent synthetic samples with temporal/spectral constraints; (2) a neuro-inspired multi-scale EEG extractor and ocular dynamics module enable efficient multimodal feature extraction; (3) a cross-modal interactive temporal attention module unifies modality interaction and temporal dependency learning. Experiments on Sleep-EDF-20 (89.1%), Sleep-EDF-78 (87.1%), and SHHS (89.7%) datasets show our method outperforms state-of-the-art approaches, validating its effectiveness and robustness.
![](https://github.com/JiangZaiUp/ISATSleepNet/blob/main/images/Architecture%20of%20the%20proposed%20model..png)

## Requirmenet
- Run our algorithm using Pytorch and CUDA 
- pip install -r requirements.txt

## Prepare datasets
The datasets used in this project are publicly available:
- Sleep-EDF-20, Sleep-EDF-78:
https://physionet.org/content/sleep-edfx/1.0.0/
- SHHS:
https://sleepdata.org/datasets/shhs/  
https://www.kaggle.com/datasets/chaimahannachi/shhs2-apnea  
Please download the datasets in advance and organize the data paths according to your local environment.
## Train Physio-GAN
Physio-GAN is used to generate physiologically plausible EEG/EOG signals for data augmentation.
```
python "./train_gan.py" --data_dir  "path/to/dataset" --ann_dir  "path/to/label" --model_dir  "path/to/save/model" --training_samples_dir  "path/to/save/training/samples" --select_ch  "EEG/EOG" --file_list  "path/to/select/files" --epochs  epoch_number --batch_size  batch_size_number --log_file  "path/to/log"
```
## Data Augmentation with Physio-GAN
After training Physio-GAN, it can be used to augment the dataset.
```
python "./augment_with_gan.py" --data_dir  "path/to/dataset" --ann_dir  "path/to/label" --output_dir  "path/to/augment_dataset"  --select_ch  "EEG/EOG" --gan_model_dir  "path/to/Physio-GAN" 
```
Comparison of real and Physio-GAN–synthesized signals in the time domain and power spectral density (PSD).
![](https://github.com/JiangZaiUp/ISATSleepNet/blob/main/images/Comparison%20of%20real%20and%20generated%20samples.png)
## Train and Validate ISATSleepNet
ISATSleepNet is trained and evaluated via cross-validation.
```
python "./ISATSleepNet.py" --log_dir  "path/to/log" --data_folder_train  "path/to/augment_dataset" --data_folder_test  "path/to/original_dataset" --fold_dir  "path/to/fold/record"  --start_fold  start_number --device_ids  GPU_index --num_epoch  epoch_number
```
To validate the effectiveness of the proposed method, we compared our model with several representative sleep staging approaches based on CNN, RNN, and Transformer architectures. The comparison results are shown below.
![](https://github.com/JiangZaiUp/ISATSleepNet/blob/main/images/Performance%20Comparison%20Between%20Previous%20Works%20on%20the%20Experimental%20Databases.png)
## Test and Visualization
To evaluate the model qualitatively, the predicted hypnogram is visualized and compared with the ground truth.
```
python "./ISATSleepNet_test.py" --model_path  "path/to/ISATSleepNet" --npz_file_path  "path/to/npz_file" --output_dir  "path/to/save_directory" --device  GPU_index
```
A randomly selected subject from the SHHS dataset was used for sleep staging, and the resulting hypnogram is shown below.
![](https://github.com/JiangZaiUp/ISATSleepNet/blob/main/images/True%20and%20predicted%20hypnogram.png)
