# -*- coding: utf-8 -*-

# 
# 학습 데이터에서 특징량의 평균과 표준편차를 구합니다.
# 

# 수치 연산 모듈(numpy)을 임포트
import numpy as np

# os, sys 모듈을 임포트
import os
import sys

# 
# 메인 함수
# 
if __name__ == "__main__":
    
    # 
    # 설정 시작
    # 

    # 두 가지 특징량
    #feature_list = ['fbank', 'mfcc']
    feature_list = ['mfcc']

    # 각 특징량[fbank, mfcc]에 대해 실행
    for feature in feature_list:
        # 각 특징량 파일의 리스트와
        # 평균 및 표준편차 계산 결과의 출력 위치
        train_feat_scp = \
            './%s/train/feats.scp' % (feature)
        train_out_dir = \
            './%s/train' % (feature)

        # 특징량 파일 리스트와 출력 위치를 리스트로 만듦
        feat_scp_list = [train_feat_scp] 
        out_dir_list = [train_out_dir]

        # 각 세트에 대해 처리를 실행
        for (feat_scp, out_dir) in \
                zip(feat_scp_list, out_dir_list):

            print('Input feat_scp: %s' % (feat_scp))

            # 출력 디렉토리가 없으면 생성
            os.makedirs(out_dir, exist_ok=True)

            # 특징량의 평균과 분산
            feat_mean = None
            feat_var = None
            # 총 프레임 수
            total_frames = 0

            # 특징량 리스트를 엶
            with open(feat_scp, mode='r') as file_feat:
                # 특징량 리스트를 한 줄씩 읽음
                for i, line in enumerate(file_feat):
                    # ...
                    parts = line.split()
                    # 0번째가 발화 ID
                    utterance_id = parts[0]
                    # 1번째가 특징량 파일 경로
                    feat_path = parts[1]
                    # 2번째가 프레임 수
                    num_frames = int(parts[2])
                    # 3번째가 차원 수
                    num_dims = int(parts[3])
                                 
                    # 특징량 데이터를 특징량 파일에서 읽음
                    feature = np.fromfile(feat_path,
                                          dtype=np.float32)

                    # 읽은 시점에서 feature는 한 행의
                    # 벡터(요소 수=프레임 수*차원 수)로
                    # 저장됨, 이를 프레임 수 x 차원 수의
                    # 행렬 형식으로 변환
                    feature = feature.reshape(num_frames, 
                                              num_dims)
     
                    # 첫 파일을 처리할 때
                    # 평균과 분산을 초기화
                    if i == 0:
                        feat_mean = np.zeros(num_dims, np.float32)
                        feat_var = np.zeros(num_dims, np.float32)

                    # 총 프레임 수를 더함
                    total_frames += num_frames
                    # 특징량 벡터의 프레임 총합을 더함
                    feat_mean += np.sum(feature, 
                                        axis=0)
                    # 특징량 벡터의 제곱 프레임 총합을 더함
                    feat_var += np.sum(np.power(feature,2), 
                                       axis=0)
            
            # 총 프레임 수로 나누어 평균 벡터 계산
            feat_mean /= total_frames
            # 분산 벡터 계산
            feat_var = (feat_var / total_frames) \
                       - np.power(feat_mean,2)
            # 제곱근을 취해 표준편차 벡터를 산출
            feat_std = np.sqrt(feat_var)

            # 파일에 씀
            out_file = os.path.join(out_dir, 'mean_std.txt')
            print('Output file: %s' % (out_file))
            with open(out_file, mode='w') as file_o:
                # 평균 벡터 작성
                file_o.write('mean\n')
                for i in range(np.size(feat_mean)):
                    file_o.write('%e ' % (feat_mean[i]))
                file_o.write('\n')
                # 표준편차 벡터 작성
                file_o.write('std\n')
                for i in range(np.size(feat_std)):
                    file_o.write('%e ' % (feat_std[i]))
                file_o.write('\n')

