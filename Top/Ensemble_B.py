import argparse
import pickle
import numpy as np
from tqdm import tqdm

def predict_with_weights(weights, r_values):
    weighted_sum = np.zeros_like(r_values[0])
    for r_val, weight in zip(r_values, weights):
        weighted_sum += r_val * weight
    return weighted_sum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default = 'V1')
    # parser.add_argument('--new_test_r1_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/infogcn/FR_Head_1/best_score.pkl')
    # parser.add_argument('--new_test_r2_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/infogcn/FR_Head_2/best_score.pkl')
    # parser.add_argument('--new_test_r3_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/infogcn/FR_Head_6/best_score.pkl')
    # parser.add_argument('--new_test_r4_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/infogcn/32frame/best_score.pkl')
    # parser.add_argument('--new_test_r5_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/infogcn/128frame/best_score.pkl')
    # parser.add_argument('--new_test_r6_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/infogcn/focalloss/best_score.pkl')
    # parser.add_argument('--new_test_r7_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/mixformer/_1/best_score.pkl')
    # parser.add_argument('--new_test_r8_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/mixformer/_2/best_score.pkl')
    # parser.add_argument('--new_test_r9_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/mixformer/_6/best_score.pkl')
    # parser.add_argument('--new_test_r10_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/mixformer/angle_1/best_score.pkl')
    # parser.add_argument('--new_test_r11_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/mixformer/motion_1/best_score.pkl')
    # parser.add_argument('--new_test_r12_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/ctr_B/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r13_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/ctr_BM/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r14_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/ctr_J/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r15_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/ctr_JM/epoch1_test_score.pkl') #TE-GCN
    # parser.add_argument('--new_test_r16_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r17_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r18_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/tdgcn_V1_B/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r19_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CTRGCN/tdgcn_V1_J/epoch1_test_score.pkl') #TE-GCN

    # parser.add_argument('--new_test_r20_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/sttformer/bone/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r21_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/sttformer/joint/epoch1_test_score.pkl') #TE-GCN
    # parser.add_argument('--new_test_r22_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/sttformer/motion/epoch1_test_score.pkl')

    # parser.add_argument('--new_test_r23_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CDGCN/bone/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r24_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CDGCN/bone_motion/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r25_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CDGCN/joint/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r26_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/CDGCN/joint_motion/epoch1_test_score.pkl')

    # parser.add_argument('--new_test_r23_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Train/tegcn/joint_motion/epoch1_test_score.pkl')


    # parser.add_argument('--new_test_r16_Score', default = '/home/zjl_laoshi/xiaoke/TE-GCN/work_dir/bone_A/epoch1_test_score.pkl')  
    # parser.add_argument('--new_test_r17_Score', default = '/home/zjl_laoshi/xiaoke/TE-GCN/work_dir/jm_A/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r18_Score', default = '/home/zjl_laoshi/xiaoke/TE-GCN/work_dir/bm_A/epoch1_test_score.pkl') 

    parser.add_argument('--new_test_r1_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/infogcn/FR_Head_1/best_score.pkl')
    parser.add_argument('--new_test_r2_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/infogcn/FR_Head_2/best_score.pkl')
    parser.add_argument('--new_test_r3_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/infogcn/FR_Head_6/best_score.pkl')
    parser.add_argument('--new_test_r4_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/infogcn/32frame/best_score.pkl')
    parser.add_argument('--new_test_r5_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/infogcn/128frame/best_score.pkl')
    parser.add_argument('--new_test_r6_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/infogcn/focalloss/best_score.pkl')
    parser.add_argument('--new_test_r7_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/mixformer/_1/best_score.pkl')
    parser.add_argument('--new_test_r8_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/mixformer/_2/best_score.pkl')
    parser.add_argument('--new_test_r9_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/mixformer/_6/best_score.pkl')
    parser.add_argument('--new_test_r10_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/mixformer/angle_1/best_score.pkl')
    parser.add_argument('--new_test_r11_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/mixformer/motion_1/best_score.pkl')
    parser.add_argument('--new_test_r12_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/ctr_B/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r13_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/ctr_BM/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r14_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/ctr_J/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r15_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/ctr_JM/epoch1_test_score.pkl') #TE-GCN
    parser.add_argument('--new_test_r16_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r17_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r18_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r19_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CTRGCN/tdgcn_V1_J/epoch1_test_score.pkl') #TE-GCN

    parser.add_argument('--new_test_r20_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/sttformer/bone/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r21_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/sttformer/joint/epoch1_test_score.pkl') #TE-GCN
    parser.add_argument('--new_test_r22_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/sttformer/motion/epoch1_test_score.pkl')

    parser.add_argument('--new_test_r23_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CDGCN/bone/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r24_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CDGCN/bone_motion/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r25_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CDGCN/joint/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r26_Score', default = '/home/coop/quminghaonan/quminghaonan/ensemble/Test/CDGCN/joint_motion/epoch1_test_score.pkl')


    # parser.add_argument('--new_test_r24_Score', default = '/home/zvlab10/xiaoke/quminghaonan/UAV-SAR/infogcn/test/_6/epoch1_test_score.pkl')
    # parser.add_argument('--new_test_r25_Score', default = '/home/zvlab10/xiaoke/quminghaonan/UAV-SAR/infogcn/test/32frame_1/epoch1_test_score.pkl') #TE-GCN
    # parser.add_argument('--new_test_r26_Score', default = '/home/zvlab10/xiaoke/quminghaonan/UAV-SAR/infogcn/test/128frame_1/epoch1_test_score.pkl') 
    # parser.add_argument('--new_test_r27_Score', default = '/home/zvlab10/xiaoke/quminghaonan/UAV-SAR/infogcn/test/focalloss_1/epoch1_test_score.pkl')
    arg = parser.parse_args()
    
    # Load new test data scores
    scores = []
    for i in range(1, 27):
        with open(getattr(arg, f'new_test_r{i}_Score'), 'rb') as f:
            scores.append(list(pickle.load(f).items()))

    # Assume we have a function to get accuracies for each model
    # accuracies = [0.7, 0.7, 0.2, 0.2, 0.2, 0.2, 0.7, 0.7, 0.2, 0.2, 
    #               0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.2, 0.2, 0.82, 0.82, 
    #               0.82, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # 
    # accuracies = [1.5, 1.5, 1.8, 0.1, 1.2, 0.8, 1.2, 1.2, 1.2, 0.8, 
    #               0.1, 1, 0.1, 1, 0.1, 2, 1.8, 1, 0.2, 2,
    #               1, 1, 0.5, 0.5]  # Example accuracies  75%

    accuracies = [1.5, 1.5, 1.5, 1, 1.5, 0, 1, 1, 1, 1,
                  0.5, 1.5, 1, 2, 1, 0.5, 0.5, 0.5, 0.5, 1,
                  1.5, 0.5, 1.5, 1, 1.5, 1]  # Example accuracies  75%
    # accuracies = [0.7035, 0.6585, 0.4825, 0.3715, 0.494, 0.486, 0.7105, 0.704, 0.7155, 0.703, 0.701, 0.7, 0.7225]  # Example accuracies   74.6%

    # Normalize accuracies to sum to 1 for weights
    weights = np.array(accuracies) / sum(accuracies)

    # Apply weights on the new test set
    predictions_new = []
    for i in range(len(scores[0])):
#        print(len(scores[1][1]))
        r_values_new = [scores[j][i][1] for j in range(len(scores))]
        prediction = predict_with_weights(weights, r_values_new)
        predictions_new.append(prediction)

    # Save the new test set predictions
    np.save('new_test_predictions.npy', predictions_new)
