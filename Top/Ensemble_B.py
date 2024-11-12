import argparse
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

# Global variables to track the best accuracy and predictions
best_acc = -1
best_predictions = []
best_weights = []

def objective(weights):
    global best_acc, best_predictions, best_weights
    right_num = total_num = 0
    predictions = []

    for i in tqdm(range(len(label))):
        l = label[i]
        r_values = [r[i][1] for r in [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10,
                                      r11, r12, r13, r14, r15, r16, r17, r18, r19, r20,
                                      r21, r22, r23, r24, r25, r26]]
        # r_values = [r[i][1] for r in [r7, r8]]
        
        r = sum(r_val * weight for r_val, weight in zip(r_values, weights))
        r = np.argmax(r)
        predictions.append(r)
        right_num += int(r == int(l))
        total_num += 1
        
    acc = right_num / total_num

    # Save the predictions for the current weight configuration
    # np.save('predictions.npy', predictions)
    
    # Save the best predictions and accuracy
    if acc > best_acc:
        best_acc = acc
        best_predictions = predictions.copy()
        best_weights = weights

    print(f'Current accuracy: {acc}')
    return -acc

def predict_with_weights(weights, r_values):
    return sum(r_val * weight for r_val, weight in zip(r_values, weights))
    # r = sum(r_val * weight for r_val, weight in zip(r_values, weights))
    # return np.argmax(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default = 'V1')
    parser.add_argument('--new_train_r1_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/infogcn/FR_Head_1/best_score.pkl')
    parser.add_argument('--new_train_r2_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/infogcn/FR_Head_2/best_score.pkl')
    parser.add_argument('--new_train_r3_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/infogcn/FR_Head_6/best_score.pkl')
    parser.add_argument('--new_train_r4_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/infogcn/32frame/best_score.pkl')
    parser.add_argument('--new_train_r5_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/infogcn/128frame/best_score.pkl')
    parser.add_argument('--new_train_r6_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/infogcn/focalloss/best_score.pkl')
    parser.add_argument('--new_train_r7_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/mixformer/_1/best_score.pkl')
    parser.add_argument('--new_train_r8_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/mixformer/_2/best_score.pkl')
    parser.add_argument('--new_train_r9_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/mixformer/_6/best_score.pkl')
    parser.add_argument('--new_train_r10_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/mixformer/angle_1/best_score.pkl')
    parser.add_argument('--new_train_r11_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/mixformer/motion_1/best_score.pkl')
    parser.add_argument('--new_train_r12_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r13_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r14_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r15_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_JM/epoch1_test_score.pkl') #TE-GCN
    parser.add_argument('--new_train_r16_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r17_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r18_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r19_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/tdgcn_V1_J/epoch1_test_score.pkl') #TE-GCN

    parser.add_argument('--new_train_r20_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/sttformer/bone/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r21_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/sttformer/joint/epoch1_test_score.pkl') #TE-GCN
    parser.add_argument('--new_train_r22_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/sttformer/motion/epoch1_test_score.pkl')

    parser.add_argument('--new_train_r23_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/cdgcn/bone/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r24_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/cdgcn/bone_motion/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r25_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/cdgcn/joint/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r26_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/cdgcn/joint_motion/epoch1_test_score.pkl')

    parser.add_argument('--new_train_r27_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/infogcn/angle_1/best_score.pkl')
    parser.add_argument('--new_train_r28_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/mixformer/motion_6/best_score.pkl')
    parser.add_argument('--new_train_r29_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_AB/epoch1_test_score.pkl')
    parser.add_argument('--new_train_r30_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Train/MixGCN/ctrgcn_V1_AJ/epoch1_test_score.pkl')

    #B
    parser.add_argument('--new_test_r1_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/infogcn/FR_Head_1/best_score.pkl')
    parser.add_argument('--new_test_r2_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/infogcn/FR_Head_2/best_score.pkl')
    parser.add_argument('--new_test_r3_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/infogcn/FR_Head_6/best_score.pkl')
    parser.add_argument('--new_test_r4_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/infogcn/32frame/best_score.pkl')
    parser.add_argument('--new_test_r5_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/infogcn/128frame/best_score.pkl')
    parser.add_argument('--new_test_r6_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/infogcn/focalloss/best_score.pkl')
    parser.add_argument('--new_test_r7_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/mixformer/_1/best_score.pkl')
    parser.add_argument('--new_test_r8_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/mixformer/_2/best_score.pkl')
    parser.add_argument('--new_test_r9_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/mixformer/_6/best_score.pkl')
    parser.add_argument('--new_test_r10_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/mixformer/angle_1/best_score.pkl')
    parser.add_argument('--new_test_r11_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/mixformer/motion_1/best_score.pkl')
    parser.add_argument('--new_test_r12_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctr_B/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r13_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctr_BM/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r14_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctr_J/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r15_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctr_JM/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r16_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctrgcn_V1_B_3D/epoch1_test_score.pkl') 
    parser.add_argument('--new_test_r17_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r18_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r19_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/tdgcn_V1_J/epoch1_test_score.pkl')
     #TE-GCN
    parser.add_argument('--new_test_r20_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/sttformer/bone/epoch1_test_score.pkl') 
    parser.add_argument('--new_test_r21_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/sttformer/joint/epoch1_test_score.pkl') #TE-GCN
    parser.add_argument('--new_test_r22_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/sttformer/motion/epoch1_test_score.pkl')

    parser.add_argument('--new_test_r23_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/cdgcn/bone/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r24_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/cdgcn/bone_motion/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r25_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/cdgcn/joint/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r26_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/cdgcn/joint_motion/epoch1_test_score.pkl')

    parser.add_argument('--new_test_r27_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/infogcn/angle_1/best_score.pkl')
    parser.add_argument('--new_test_r28_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/mixformer/motion_6/best_score.pkl')
    parser.add_argument('--new_test_r29_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctr_AB/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r30_Score', default = '/home/zjl_laoshi/quminghaonan/quminghaonan1/output/Test/MixGCN/ctrAJ/epoch1_test_score.pkl')
    arg = parser.parse_args()
    
    benchmark = arg.benchmark
    if benchmark == 'V1':
        npz_data = np.load('/home/zjl_laoshi/quminghaonan/dataset/data/val_label.npy')
        label = npz_data
    else:
        assert benchmark == 'V2'
        npz_data = np.load('/home/zjl_laoshi/quminghaonan/dataset/data/val_label.npy')
        label = npz_data

    # Load all scores
    with open(arg.new_train_r1_Score, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.new_train_r2_Score, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(arg.new_train_r3_Score, 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(arg.new_train_r4_Score, 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(arg.new_train_r5_Score, 'rb') as r5:
        r5 = list(pickle.load(r5).items())
        
    with open(arg.new_train_r6_Score, 'rb') as r6:
        r6 = list(pickle.load(r6).items())
    
    with open(arg.new_train_r7_Score, 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(arg.new_train_r8_Score, 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    with open(arg.new_train_r9_Score, 'rb') as r9:
        r9 = list(pickle.load(r9).items())

    with open(arg.new_train_r10_Score, 'rb') as r10:
        r10 = list(pickle.load(r10).items())

    with open(arg.new_train_r11_Score, 'rb') as r11:
        r11 = list(pickle.load(r11).items())
        
    with open(arg.new_train_r12_Score, 'rb') as r12:
        r12 = list(pickle.load(r12).items())
    
    with open(arg.new_train_r13_Score, 'rb') as r13:
        r13 = list(pickle.load(r13).items())

    with open(arg.new_train_r14_Score, 'rb') as r14:
        r14 = list(pickle.load(r14).items())

    with open(arg.new_train_r15_Score, 'rb') as r15:
        r15 = list(pickle.load(r15).items())

    with open(arg.new_train_r16_Score, 'rb') as r16:
        r16 = list(pickle.load(r16).items())

    with open(arg.new_train_r17_Score, 'rb') as r17:
        r17 = list(pickle.load(r17).items())

    with open(arg.new_train_r18_Score, 'rb') as r18:
        r18 = list(pickle.load(r18).items())

    with open(arg.new_train_r19_Score, 'rb') as r19:
        r19 = list(pickle.load(r19).items())
    
    with open(arg.new_train_r20_Score, 'rb') as r20:
        r20 = list(pickle.load(r20).items())

    with open(arg.new_train_r21_Score, 'rb') as r21:
        r21 = list(pickle.load(r21).items())

    with open(arg.new_train_r22_Score, 'rb') as r22:
        r22 = list(pickle.load(r22).items())

    with open(arg.new_train_r23_Score, 'rb') as r23:
        r23 = list(pickle.load(r23).items())

    with open(arg.new_train_r24_Score, 'rb') as r24:
        r24 = list(pickle.load(r24).items())

    with open(arg.new_train_r25_Score, 'rb') as r25:
        r25 = list(pickle.load(r25).items())
    
    with open(arg.new_train_r26_Score, 'rb') as r26:
        r26 = list(pickle.load(r26).items())

    with open(arg.new_train_r27_Score, 'rb') as r27:
        r27 = list(pickle.load(r27).items())

    with open(arg.new_train_r28_Score, 'rb') as r28:
        r28 = list(pickle.load(r28).items())

    with open(arg.new_train_r29_Score, 'rb') as r29:
        r29 = list(pickle.load(r29).items())
    
    with open(arg.new_train_r30_Score, 'rb') as r30:
        r30 = list(pickle.load(r30).items())

    # space = [(0.2, 1.2) for i in range(15)]
    space = [(0.1, 1.5) for i in range(31)]
    result = gp_minimize(objective, space, n_calls=50, random_state=1)
    
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))
    
    # Save the best predictions to a file
    np.save('best_predictions.npy', best_predictions)

    # Apply weights to a new test dataset
    # best_weights = result.x
    
    # Load new test data scores
    with open(arg.new_test_r1_Score, 'rb') as r1:
        r1_new = list(pickle.load(r1).items())
    
    with open(arg.new_test_r2_Score, 'rb') as r2:
        r2_new = list(pickle.load(r2).items())
    
    with open(arg.new_test_r3_Score, 'rb') as r3:
        r3_new = list(pickle.load(r3).items())
    
    with open(arg.new_test_r4_Score, 'rb') as r4:
        r4_new = list(pickle.load(r4).items())
    
    with open(arg.new_test_r5_Score, 'rb') as r5:
        r5_new = list(pickle.load(r5).items())
    
    with open(arg.new_test_r6_Score, 'rb') as r6:
        r6_new = list(pickle.load(r6).items())
    
    with open(arg.new_test_r7_Score, 'rb') as r7:
        r7_new = list(pickle.load(r7).items())
    
    with open(arg.new_test_r8_Score, 'rb') as r8:
        r8_new = list(pickle.load(r8).items())
    
    with open(arg.new_test_r9_Score, 'rb') as r9:
        r9_new = list(pickle.load(r9).items())
    
    with open(arg.new_test_r10_Score, 'rb') as r10:
        r10_new = list(pickle.load(r10).items())
    
    with open(arg.new_test_r11_Score, 'rb') as r11:
        r11_new = list(pickle.load(r11).items())
    
    with open(arg.new_test_r12_Score, 'rb') as r12:
        r12_new = list(pickle.load(r12).items())
    
    with open(arg.new_test_r13_Score, 'rb') as r13:
        r13_new = list(pickle.load(r13).items())
    
    with open(arg.new_test_r14_Score, 'rb') as r14:
        r14_new = list(pickle.load(r14).items())
    
    with open(arg.new_test_r15_Score, 'rb') as r15:
        r15_new = list(pickle.load(r15).items())

    with open(arg.new_test_r16_Score, 'rb') as r16:
        r16_new = list(pickle.load(r16).items())
    
    with open(arg.new_test_r17_Score, 'rb') as r17:
        r17_new = list(pickle.load(r17).items())
    
    with open(arg.new_test_r18_Score, 'rb') as r18:
        r18_new = list(pickle.load(r18).items())

    with open(arg.new_test_r19_Score, 'rb') as r19:
        r19_new = list(pickle.load(r19).items())
    
    with open(arg.new_test_r20_Score, 'rb') as r20:
        r20_new = list(pickle.load(r20).items())

    with open(arg.new_test_r21_Score, 'rb') as r21:
        r21_new = list(pickle.load(r21).items())

    with open(arg.new_test_r22_Score, 'rb') as r22:
        r22_new = list(pickle.load(r22).items())

    with open(arg.new_test_r23_Score, 'rb') as r23:
        r23_new = list(pickle.load(r23).items())

    with open(arg.new_test_r24_Score, 'rb') as r24:
        r24_new = list(pickle.load(r24).items())

    with open(arg.new_test_r25_Score, 'rb') as r25:
        r25_new = list(pickle.load(r25).items())
    
    with open(arg.new_test_r26_Score, 'rb') as r26:
        r26_new = list(pickle.load(r26).items())

    with open(arg.new_test_r27_Score, 'rb') as r27:
        r27_new = list(pickle.load(r27).items())

    with open(arg.new_test_r28_Score, 'rb') as r28:
        r28_new = list(pickle.load(r28).items())

    with open(arg.new_test_r29_Score, 'rb') as r29:
        r29_new = list(pickle.load(r29).items())
    
    with open(arg.new_test_r30_Score, 'rb') as r30:
        r30_new = list(pickle.load(r30).items())

    # with open(arg.new_test_r27_Score, 'rb') as r27:
    #     r27_new = list(pickle.load(r27).items())
    
    # Apply weights on the new test set
    predictions_new = []
    for i in range(len(r1_new)):
        r_values_new = [r[i][1] for r in [r1_new, r2_new, r3_new, r4_new, r5_new, r6_new, r7_new, r8_new, r9_new, r10_new,
                                          r11_new, r12_new, r13_new, r14_new, r15_new, r16_new, r17_new, r18_new, r19_new, r20_new,
                                          r21_new, r22_new, r23_new, r24_new, r25_new, r26_new, r27_new, r28_new, r29_new, r30_new]]
        # r_values_new = [r[i][1] for r in [r7_new, r8_new]]
        prediction = predict_with_weights(best_weights, r_values_new)
        predictions_new.append(prediction)
    # Save the new test set predictions
    np.save('new_test_predictions.npy', predictions_new)
