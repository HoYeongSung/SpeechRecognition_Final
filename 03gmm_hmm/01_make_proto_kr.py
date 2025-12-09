# -*- coding: utf-8 -*-

# 
# Monophone-HMM 정의 파일을 생성합니다.
# 생성하는 HMM은 left-to-right 형식입니다.
# 혼합 수는 1입니다. 대각 공분산 행렬을 가정합니다.
# 

# hmmfunc.py에서 MonoPhoneHMM 클래스를 임포트
from hmmfunc import MonoPhoneHMM

# os 모듈을 임포트
import os

# 
# 메인 함수
# 
if __name__ == "__main__":

    # 음소 리스트
    phone_list_file = \
        './exp/data/train/phone_list'

    '''HMM 상태 수 최적화 시 건들 곳'''
    # 각 음소 HMM의 상태 수
    NUM_STATES = 3

    # 입력 특징의 차원 수
    # 여기서는 MFCC를 사용하기 때문에,
    # MFCC의 차원 수를 입력
    num_dims = 13

    # 자기 루프 확률의 초기값
    prob_loop = 0.7

    '''HMM 상태 수 최적화-전역변수로 변환'''
    # 출력 폴더
    out_dir = f'./exp/model_{NUM_STATES}state_1mix'

    # 
    # 처리 시작
    # 
    # 출력 디렉토리가 없으면 생성
    os.makedirs(out_dir, exist_ok=True)

    # 음소 리스트 파일을 열고, phone_list에 저장
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 음소 리스트 파일에서 음소를 가져옴
            phone = line.split()[0]
            # 음소 리스트의 끝에 추가
            phone_list.append(phone)

    # MonoPhoneHMM 클래스를 호출
    hmm = MonoPhoneHMM()

    # HMM 프로토타입을 생성
    hmm.make_proto(phone_list, NUM_STATES,
                   prob_loop, num_dims)

    # HMM 프로토타입을 JSON 형식으로 저장
    hmm.save_hmm(os.path.join(out_dir, 'hmmproto'))

