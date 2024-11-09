# CD-GCN

## Result

|      Method      |   CS1%   |   CS2%   | 
|:----------------:|:--------:|:--------:|
|      ST-GCN      |   38.9   |   63.9   |
|       AGCN       |   40.0   |   66.2   |
|    Shift-GCN     |   40.8   |   66.5   |
|      TE-GCN      |   42.5   |   68.2   |
|     FR-AGCN      |   44.0   |   69.5   |
| **CD-GCN(Ours)** | **46.9** | **74.2** |

## Data Preparation

https://drive.google.com/drive/folders/19wGrIx63Ki3-6rBPqR8TxC1vjR8tCKkv?usp=sharing

You can download the processed dataset from Google Drive above, or you can process the dataset yourself by using the following methods.

### Download dataset

1. Request dataset here: https://sutdcv.github.io/uav-human-web/
2. Download the ``Skeleton.zip``
3. More information about UAV-Human, please refer [UAV-Human](https://github.com/SUTDCV/UAV-Human/)

### Data Processing

#### Directory Structure

1. Extract the downloaded data ``Skeleton.zip`` into ``data`` folder.
2. Expected directory in  ``data`` folder.
    ```
    └───Skeleton
        ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
        ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
        └───...
    ```

#### Generating Data

1. Run the following scripts to split the files into the train and the test set respectively:
   ```
   python split_v1.py 
   python split_v2.py 
   ```

2. Run the following script to preprocess the files:
   ```
   python generate_data.py --data_path data/v1 
   python generate_data.py --data_path data/v2 
   ```

3. Expected directory in `data/v1`
   ```
   └───data/v1
       ├───train
           ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
           ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
           └───...
       ├───test
           ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
           ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
           └───...
       ├───train_label.pkl
       ├───train_data.npy
       ├───test_data.npy
       └───test_label.pkl
   ```
   `data/v2` has the similar structure.

4. Move  ``.pkl`` files and ``.npy`` files from the ``data/v1`` and ``data/v2`` folders into the ``data/uav/xsub1``
   and ``data/uav/xsub2`` folders, respectively.
   Your ``data/uav`` should be like this:
   ```
   uav
   ___ xsub1
       ___ test_data.npy
       ___ test_label.pkl
       ___ train_data.npy
       ___ train_label.pkl
   ___ xsub2
       ___ test_data.npy
       ___ test_label.pkl
       ___ train_data.npy
       ___ train_label.pkl
   
   ```

## Training & Testing

### Training

You can train your model using the scripts:

```
sh scripts/TRAIN_V1.sh
sh scripts/TRAIN_V2.sh
```

### TEST

- To test the trained models saved in <work_dir>, run the following command:

```
sh scripts/EVAL_V1.sh
sh scripts/EVAL_V2.sh
```

- To ensemble the results of different modalities, run

```
# Example: ensemble six modalities of CDGCN on UAV/xsub1
python ensemble_6s.py --dataset UAV/xsub1 --joint work_dir/xsub1_0/epoch36_test_score.pkl --bone work_dir/xsub1_1/epoch36_test_score.pkl --joint-motion work_dir/xsub1_4/epoch36_test_score.pkl --bone-motion work_dir/xsub1_5/epoch36_test_score.pkl --prompt work_dir/xsub1_2/epoch43_test_score.pkl --prompt2 work_dir/xsub1_3/epoch37_test_score.pkl
```

```
# Example: ensemble six modalities of CDGCN on UAV/xsub2
python ensemble_6s.py --dataset UAV/xsub2 --joint work_dir/2103/epoch33_test_score.pkl --bone work_dir/2104/epoch34_test_score.pkl --joint-motion work_dir/vel_01/epoch34_test_score.pkl --bone-motion work_dir/vel_0/epoch34_test_score.pkl --prompt work_dir/vel_02/epoch34_test_score.pkl --prompt2 work_dir/vel_03/epoch32_test_score.pkl
```
