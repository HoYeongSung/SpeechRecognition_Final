# -*- coding: utf-8 -*-

# 
# GMM-HMM을 학습합니다.
# 

# hmmfunc.py에서 MonoPhoneHMM 클래스를 임포트
from hmmfunc import MonoPhoneHMM

# 수치 연산 모듈(numpy)을 임포트
import numpy as np

# os, sys 모듈을 임포트
import sys
import os
import time

# 
# 메인 함수
# 
if __name__ == "__main__":

    # 학습용 HMM 파일
    base_hmm = './exp/model_3state_1mix/10.hmm'

    # 훈련 데이터 특징량 리스트 파일
    feat_scp = \
        '../01compute_features/mfcc/train/feats.scp'

    # 훈련 데이터의 라벨 파일
    label_file = \
        './exp/data/train/text_int'

    # 학습 결과를 저장할 폴더
    work_dir = './exp/gmmhmm'

    # 업데이트 횟수
    num_iter = 10

    # 혼합 수를 늘리는 횟수
    # 늘릴 때마다 혼합 수는 2배가 됩니다
    # 최종 혼합 수는 2^(mixup_time)이 됩니다
    # 업데이트 횟수는 num_iter*(mixup_time+1)
    mixup_time = 1

    # 학습에 사용할 발화 수
    # 실제로는 모든 발화를 사용하지만, 시간이 걸리기 때문에
    # 이 프로그램에서는 일부 발화만 사용
    num_utters = 1050

    # 
    # 처리 시작
    # 

    '''CPU 병렬 처리를 위한 수정 사항'''
    # CPU 병렬 처리를 위한 워커 수
    num_workers = 4

    # MonoPhoneHMM 클래스를 호출
    hmm = MonoPhoneHMM()

    # 학습 전 HMM을 읽어들임
    hmm.load_hmm(base_hmm)

    # 학습용 HMM 파일의 상태 수를 얻음
    num_states = hmm.num_states

    # 학습용 HMM 파일의 혼합 수를 얻음
    num_mixture = hmm.num_mixture

    # 라벨 파일을 열어 발화 ID별
    # 라벨 정보를 얻음
    label_list = {}
    with open(label_file, mode='r') as f:
        for line in f:
            # 0열은 발화 ID
            utt = line.split()[0]
            # 1열 이후는 라벨
            lab = line.split()[1:]
            # 각 요소는 문자열로 읽히므로,
            # 정수 값으로 변환
            lab = np.int64(lab)
            # label_list에 등록
            label_list[utt] = lab

    # 특징량 리스트 파일을 열고,
    # 발화 ID별 특징량 파일 경로를 얻음
    feat_list = {}
    with open(feat_scp, mode='r') as f:
        # 특징량 경로를 feat_list에 추가
        # 이때, 학습에 사용할 발화 수만큼만 추가
        # (全てのデータを学習に用いると時間がかかるため)
        for n, line in enumerate(f):
            if n >= num_utters:
                break
            # 0열은 발화 ID
            utt = line.split()[0]
            # 1열은 파일 경로
            ff = line.split()[1]
            # 3열은 차원 수
            nd = int(line.split()[3])
            # 발화 ID가 label_에 없으면 에러
            if not utt in label_list:
                sys.stderr.write(\
                    '%s does not have label\n' % (utt))
                exit(1)
            # 차원 수가 HMM의 차원 수와 일치하지 않으면 에러
            if hmm.num_dims != nd:
                sys.stderr.write(\
                    '%s: unexpected # dims (%d)\n'\
                    % (utt, nd))
                exit(1)
            # feat_file에 등록
            feat_list[utt] = ff


    # 
    # 학습 처리
    # 

    # 출력 디렉토리명
    model_name = 'model_%dstate_%dmix' \
        % (num_states, num_mixture)
    out_dir = os.path.join(work_dir, model_name)

    # 출력 디렉토리가 없으면 생성
    os.makedirs(out_dir, exist_ok=True)

    # 혼합 수 증가 횟수만큼 루프
    for m in range(mixup_time+1):
        # 
        # 혼합 수 증가 처리를 수행하고,
        # 새 폴더에 0.hmm이라는 이름으로 저장
        # 
        if m > 0:
            # 혼합 수 증가 처리를 수행
            hmm.mixup()
            # 혼합 수를 2배로 늘림
            num_mixture *= 2

            # 출력 디렉토리명
            model_name = 'model_%dstate_%dmix' \
                % (num_states, num_mixture)
            out_dir = os.path.join(work_dir, model_name)

            # 출력 디렉토리가 없으면 생성
            os.makedirs(out_dir, exist_ok=True)
          
            # HMM을 저장
            out_hmm = os.path.join(out_dir, '0.hmm')
            hmm.save_hmm(out_hmm)
            print('Increased mixtures %d -> %d' \
                % (num_mixture/2, num_mixture))
            print('saved model: %s' % (out_hmm))

        # 
        # 현재 GMM 혼합 수로 규정 횟수만큼 학습을 수행
        # 
        # num_iter 횟수만큼 업데이트 반복

        start_time = time.time()

        for iter in range(num_iter):
            print('%d-th iterateion' % (iter+1))
            # 학습(1회 반복)
            #   -> MonoPhoneHMM.train 내부에서 num_workers 개의 프로세스를 사용해
            #      발화별 학습을 병렬로 처리하도록 구현할 예정
            hmm.train(feat_list, label_list, num_workers=num_workers)

            # HMM 프로토타입을 JSON 형식으로 저장
            out_hmm = os.path.join(out_dir, 
                                   '%d.hmm' % (iter+1))
            # 학습한 HMM을 저장
            hmm.save_hmm(out_hmm)
            print('saved model: %s' % (out_hmm))

        # 학습 끝난 시점
        end_time = time.time()
        elapsed = end_time - start_time

        # 전체 학습 시간 출력
        print(f"Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
