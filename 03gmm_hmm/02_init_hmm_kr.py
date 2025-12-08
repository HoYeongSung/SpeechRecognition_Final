# -*- coding: utf-8 -*-

# 
# HMM 프로토타입을 읽어들이고,
# 플랫 스타트로 초기화합니다.
# 

# hmmfunc.py에서 MonoPhoneHMM 클래스를 임포트
from hmmfunc import MonoPhoneHMM

# 수치 연산 모듈(numpy)을 임포트
import numpy as np

# os 모듈을 임포트
import os

# 
# 메인 함수
# 
if __name__ == "__main__":

    # HMM 프로토타입
    hmmproto = './exp/model_3state_1mix/hmmproto'

    # 학습 데이터 특징의 평균/표준 편차 파일
    mean_std_file = \
        '../01compute_features/mfcc/train/mean_std.txt'

    # 출력 디렉토리
    out_dir = os.path.dirname(hmmproto)

    # 
    # 처리 시작
    # 

    # 출력 디렉토리가 없으면 생성
    os.makedirs(out_dir, exist_ok=True)

    # 특징의 평균/표준 편차 파일을 읽어들임
    with open(mean_std_file, mode='r') as f:
        # 모든 행 읽기
        lines = f.readlines()
        # 1행(0부터 시작)이 평균 벡터(mean),
        # 3행이 표준 편차 벡터(std)
        mean_line = lines[1]
        std_line = lines[3]
        # 공백으로 구분된 리스트로 변환
        mean = mean_line.split()
        std = std_line.split()
        # numpy 배열로 변환
        mean = np.array(mean, dtype=np.float64)
        std = np.array(std, dtype=np.float64)
        # 표준 편차를 분산으로 변환
        var = std ** 2
    
    # MonoPhoneHMM 클래스를 호출
    hmm = MonoPhoneHMM()

    # HMM 프로토타입을 읽어들임
    hmm.load_hmm(hmmproto)

    hmm.num_dims = len(mean)



    # HMM 프로토타입을 JSON 형식으로 저장
    hmm.save_hmm(os.path.join(out_dir, '0.hmm'))

    # 플랫 스타트 초기화를 실행
    hmm.flat_init(mean, var)

    # HMM 프로토타입을 JSON 형식으로 저장
    hmm.save_hmm(os.path.join(out_dir, '0.hmm'))

