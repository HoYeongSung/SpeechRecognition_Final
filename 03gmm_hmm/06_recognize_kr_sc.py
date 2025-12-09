# -*- coding: utf-8 -*-

# 
# HMM 모델로 고립 단어 인식을 수행합니다.
# 

# hmmfunc.py에서 MonoPhoneHMM 클래스를 임포트
from hmmfunc import MonoPhoneHMM

# 수치 연산 모듈(numpy)을 임포트
import numpy as np

# os, sys 모듈을 임포트
import sys
import os

# 
# 메인 함수
# 
if __name__ == "__main__":
    '''HMM 상태수 최적화'''
    NUM_STATES = 3
    # HMM 파일
    hmm_file = f'./exp/model_{NUM_STATES}state_1mix/10.hmm'
    # hmm_file = f'./exp/gmmhmm/model_{NUM_STATES}state_2mix/10.hmm'

    # 평가 데이터의 특징량 리스트 파일
    feat_scp = '../01compute_features/mfcc/test/feats.scp'

    # 사전 파일
    lexicon_file = 'sc35.dic'

    # 음소 리스트
    phone_list_file = \
        './exp/data/train/phone_list'

    # True인 경우, 문장의 시작과 끝에
    # 포즈가 있다고 가정
    insert_sil = True

    nbest = 0
    out_dir = './exp/data/test'
    result_file = out_dir + '/result.txt'
    os.makedirs(out_dir, exist_ok=True)
    # 
    # 처리 시작
    # 

    # 음소 리스트 파일을 열어 phone_list에 저장
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 음소 리스트 파일에서 음소를 가져옴
            phone = line.split()[0]
            # 음소 리스트의 끝에 추가
            phone_list.append(phone)

    # 사전 파일을 열어 단어와 음소열의 대응 리스트를 얻음
    lexicon = []
    with open(lexicon_file, mode='r') as f:
        for line in f:
            # 0열은 단어
            word = line.split()[0]
            # 1열 이후는 음소열
            phones = line.split()[1:]
            # insert_sil이 True인 경우 양 끝에 포즈를 추가
            if insert_sil:
                phones.insert(0, phone_list[0])
                phones.append(phone_list[0])
            # phone_list를 사용해 음소를 숫자로 변환
            ph_int = []
            for ph in phones:
                if ph in phone_list:
                    ph_int.append(phone_list.index(ph))
                else:
                    sys.stderr.write('invalid phone %s' % (ph))
            # 단어, 음소열, 숫자 표기의 사전으로서
            # lexicon에 추가
            lexicon.append({'word': word,
                            'pron': phones,
                            'int': ph_int})

    # MonoPhoneHMM 클래스를 호출
    hmm = MonoPhoneHMM()

    # HMM을 읽어들임
    hmm.load_hmm(hmm_file)

    # 특징량 리스트 파일을 열어
    # 발화별로 음성 인식을 수행
    with open(result_file, mode='w') as f_out:
        with open(feat_scp, mode='r') as f:
            for line in f:
                # 0열은 발화 ID
                utt = line.split()[0]
                # 1열은 파일 경로
                ff = line.split()[1]
                # 3열은 차원 수
                nd = int(line.split()[3])
            
                # 차원 수가 HMM의 차원 수와 일치하지 않으면 에러
                if hmm.num_dims != nd:
                    sys.stderr.write(\
                                     '%s: unexpected # dims (%d)\n'\
                                     % (utt, nd))
                    exit(1)

                # 특징량 파일을 엶
                feat = np.fromfile(ff, dtype=np.float32)
                # 프레임 수 x 차원 수 배열로 변형
                feat = feat.reshape(-1, hmm.num_dims)

                # 고립 단어 인식을 수행
                (result, detail) = hmm.recognize(feat, lexicon)

                # result에는 가장 우도가 높은 단어가 저장됨
                # detail에는 우도 순위가 저장됨
                # 결과를 출력함
                sys.stdout.write('%s %s\n' % (utt, ff))
                sys.stdout.write('Result = %s\n' % (result))
                f_out.write('%s %s\n' % (utt, ff))
                f_out.write('Result = %s\n' % (result))
                if nbest > 0:
                    sys.stdout.write('[Runking]\n')
                    #for res in detail:
                    for res in detail[:nbest]:
                        sys.stdout.write('  %s %f\n' \
                                         % (res['word'], res['score']))
       
