# -*- coding: utf-8 -*-

# 
# MFCC 특징을 계산합니다.
# 

# wav 데이터를 읽기 위한 모듈(wave)을 임포트
import wave

# 수치 연산 모듈(numpy)을 임포트
import numpy as np

# os, sys 모듈을 임포트
import os
import sys

class FeatureExtractor():
    '''특징량(FBANK, MFCC)을 추출하는 클래스
    sample_frequency: 입력 파형의 샘플링 주파수 [Hz]
    frame_length: 프레임 크기 [밀리초]
    frame_shift: 분석 간격(프레임 시프트) [밀리초]
    num_mel_bins: 멜 필터 뱅크의 개수(=FBANK 특징의 차원 수)
    num_ceps: MFCC 특징의 차원 수(0차원 포함)
    lifter_coef: 리프터링 처리의 파라미터
    low_frequency: 저주파 대역 제거의 컷오프 주파수 [Hz]
    high_frequency: 고주파 대역 제거의 컷오프 주파수 [Hz]
    dither: 디더링 처리의 파라미터(잡음의 세기) 
    '''
    # 클래스를 호출할 때 처음에 한 번 실행되는 함수
    def __init__(self, 
                 sample_frequency=16000, 
                 frame_length=25, 
                 frame_shift=10, 
                 num_mel_bins=23, 
                 num_ceps=13, 
                 lifter_coef=22, 
                 low_frequency=20, 
                 high_frequency=8000, 
                 dither=1.0):
        # 샘플링 주파수[Hz]
        self.sample_freq = sample_frequency
        # 창 길이를 밀리초에서 샘플 수로 변환
        self.frame_size = int(sample_frequency * frame_length * 0.001)
        # 프레임 시프트를 밀리초에서 샘플 수로 변환
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        # 멜 필터 뱅크의 개수
        self.num_mel_bins = num_mel_bins
        # MFCC의 차원 수 (0차 포함)
        self.num_ceps = num_ceps
        # 리프터링 매개변수
        self.lifter_coef = lifter_coef
        # 저주파수 대역 제거 절단 주파수[Hz]
        self.low_frequency = low_frequency
        # 고주파수 대력 제거 절단 주파수[Hz]
        self.high_frequency = high_frequency
        # 디더링 개수
        self.dither_coef = dither

        # FFT 포인트 수 = 창폭 이상의 2제곱
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # 멜 필터뱅크를 생성
        self.mel_filter_bank = self.MakeMelFilterBank()

        # 이산 코사인 변환(DCT) 기저 행렬을 생성
        self.dct_matrix = self.MakeDCTMatrix()

        # 리프터(lifter) 생성
        self.lifter = self.MakeLifter()


    def Herz2Mel(self, herz):
        ''' 주파수를 헤르츠에서 Mel로 변환
        '''
        return (1127.0 * np.log(1.0 + herz / 700))


    def MakeMelFilterBank(self):
        ''' Mel 필터 뱅크 생성
        '''
        # 멜 축의 최대 주파수
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        # 멜 축의 최소 주파수
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        # 최소에서 최대 주파수까지
        # Mel 축 위에서 동일 간격으로 주파수를 취한다
        mel_points = np.linspace(mel_low_freq, 
                                 mel_high_freq, 
                                 self.num_mel_bins+2)

        # 파워 스펙트럼의 차원 수 = FFT 크기/2+1
        # ※Kald의 구현에서는 나이퀴스트 주파수 성분(마지막+1)은
        # 버리고 있지만, 본 구현에서는 버리지 않고 사용하고 있음
        dim_spectrum = int(self.fft_size / 2) + 1

        # 멜 필터 뱅크(필터 수 x 스펙트럼 차원 수)
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            # 삼각 필터의 왼쪽 끝, 중심, 오른쪽 끝의 멜 주파수
            left_mel = mel_points[m]
            center_mel = mel_points[m+1]
            right_mel = mel_points[m+2]
            # 파워 스펙트럼의 각 bin에 대응하는 가중치를 계산
            for n in range(dim_spectrum):
                # 각 빈에 해당하는 헤르츠 축 주파수를 계산
                freq = 1.0 * n * self.sample_freq/2 / dim_spectrum
                # Mel 주파수로 변환
                mel = self.Herz2Mel(freq)
                # 그 빈이 삼각 필터 범위에 들어가면 가중치를 계산
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel-mel) / (right_mel-center_mel)
                    mel_filter_bank[m][n] = weight
         
        return mel_filter_bank

    
    def ExtractWindow(self, waveform, start_index, num_samples):
        '''
        1프레임 분량의 파형 데이터를 추출하여 전처리하고,
        로그 파워값을 계산
        '''
        # waveform에서 1프레임 분량의 파형 추출
        window = waveform[start_index:start_index + self.frame_size].copy()

        # 디더링을 수행
        # (-dither_coef～dither_coef 사이 값을 난수로 추가)
        if self.dither_coef > 0:
            window = window \
                     + np.random.rand(self.frame_size) \
                     * (2*self.dither_coef) - self.dither_coef

        # 직류 성분을 제거
        window = window - np.mean(window)

        # 아래 처리를 실행하기 전에 파워 계산
        power = np.sum(window ** 2)
        # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
        if power < 1E-10:
            power = 1E-10
        # 로그를 취한다
        log_power = np.log(power)

        # 프리엠퍼시스(고주파 강조)
        # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window,np.array([1.0, -0.97]), mode='same')
        # numpy.convolve는 0번째 요소가 처리되지 않기
        # (window[i-1]가 없기) 때문에
        # window[0-1]을window[0]로 대체하여 처리
        window[0] -= 0.97*window[0]

        # 해밍 창을 적용
        # hamming[i] = 0.54 - 0.46 * np.cos(2*np.pi*i / (self.frame_size - 1))
        window *= np.hamming(self.frame_size)

        return window, log_power


    def ComputeFBANK(self, waveform):
        '''로그 Mel 필터뱅크특징(FBANK) 계산
        출력1: fbank_features: 로그 Mel 필터뱅크 특징
        출력2: log_power: 로그 파워 값(MFCC 추출 시에 사용)
        '''
        # 파형 데이터 총 샘플 수
        num_samples = np.size(waveform)
        # 특징값의 총 프레임 수를 계산
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        # Mel 필터 뱅크 특징
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        # 로그 파워(MFCC 특징을 구할 때 사용됨)
        log_power = np.zeros(num_frames)

        # 1프레임 마다 특징값 계산
        for frame in range(num_frames):
            # 분석 시작 위치는 프레임 번호(0에서 시작)*프레임 시프트
            start_index = frame * self.frame_shift
            # 1프레임 분량의 파형을 추출하여 전처리 수행
            # 로그 파워 값 계산
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)
            
            # 고속 푸리에 변환(FFT)을 실행
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT 결과의 오른쪽 절반(음의 주파수 성분)을 제거
            # ※Kald의 구현에서는 나이퀴스트 주파수 성분(마지막+1)은 버리고 있지만,
            # 본 구현에서는 버리지 않고 사용하고 있음
            spectrum = spectrum[:int(self.fft_size/2) + 1]

            # 파워 스펙트럼을 계산
            spectrum = np.abs(spectrum) ** 2

            # Mel 필터뱅크 계산
            fbank = np.dot(spectrum, self.mel_filter_bank.T)

            # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
            fbank[fbank<0.1] = 0.1

            # 로그를 취해서 fbank_features에 첨부
            fbank_features[frame] = np.log(fbank)

            # 로그 파워 값을 log_power에 첨부
            log_power[frame] = log_pow

        return fbank_features, log_power


    def MakeDCTMatrix(self):
        ''' 이산 코사인 변환(DCT)의 기저 행렬 작성
        '''
        N = self.num_mel_bins
        # DCT 기저행렬(기저수(=MFCC 차원수) x FBANK 차원수)
        dct_matrix = np.zeros((self.num_ceps,self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2/N) \
                    * np.cos(((2.0*np.arange(N)+1)*k*np.pi) / (2*N))

        return dct_matrix


    def MakeLifter(self):
        ''' 리프터 계산
        '''
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
        return lifter


    def ComputeMFCC(self, waveform):
        ''' MFCC 계산
        '''
        # FBANK 및 로그 파워 계산
        fbank, log_power = self.ComputeFBANK(waveform)
        
        # DCT 기저 행렬과의 곱셈으로 DCT 계산
        mfcc = np.dot(fbank, self.dct_matrix.T)

        # 리프터링
        mfcc *= self.lifter

        # MFCC의 0차원 값을 전처리하기 전에 파형 로그 파워로 치환
        mfcc[:,0] = log_power

        return mfcc

# 
# 메인 함수
# 
if __name__ == "__main__":
    
    # 
    # 설정 시작
    # 

    # 각 wav 파일의 리스트와 특징량 출력 위치
    train_wav_scp = '../data/label/train/wav.scp'
    train_out_dir = './mfcc/train'
    #dev_wav_scp = '../data/label/dev/wav.scp'
    #dev_out_dir = './mfcc/dev'
    test_wav_scp = '../data/label/test/wav.scp'
    test_out_dir = './mfcc/test'

    # 샘플링 주파수 [Hz]
    sample_frequency = 16000
    # 프레임 길이 [밀리초]
    frame_length = 25
    # 프레임 시프트 [밀리초]
    frame_shift = 10
    # 저주파수 대역 제거의 컷오프 주파수 [Hz]
    low_frequency = 20
    # 고주파수 대역 제거의 컷오프 주파수 [Hz]
    high_frequency = sample_frequency / 2
    # 로그 Mel 필터뱅크 차원수
    num_mel_bins = 23
    # MFCC 차원수
    num_ceps = 13
    # 디더링 계수
    dither=1.0

    # 난수 시드 설정(디더링 처리 결과의 재현성 확보)
    np.random.seed(seed=0)

    # 특징값 추출 클래스 불러오기
    feat_extractor = FeatureExtractor(
                       sample_frequency=sample_frequency, 
                       frame_length=frame_length, 
                       frame_shift=frame_shift, 
                       num_mel_bins=num_mel_bins, 
                       num_ceps=num_ceps,
                       low_frequency=low_frequency, 
                       high_frequency=high_frequency, 
                       dither=dither)

    # wav 파일 목록과 출력 위치를 리스트로 생성
    wav_scp_list = [train_wav_scp, 
                    #dev_wav_scp, 
                    test_wav_scp]
    out_dir_list = [train_out_dir, 
                    #dev_out_dir, 
                    test_out_dir]

    # 각 세트에 대한 처리 실행
    for (wav_scp, out_dir) in zip(wav_scp_list, out_dir_list):
        print('Input wav_scp: %s' % (wav_scp))
        print('Output directory: %s' % (out_dir))

        # 특징값 파일 경로, 프레임 수,
        # 차원수를 기록한 리스트
        feat_scp = os.path.join(out_dir, 'feats.scp')

        # 출력 디렉토리가 존재하지 않을 경우 생성
        os.makedirs(out_dir, exist_ok=True)

        # wav 목록 읽기 모드
        # 특징값 리스트를 쓰기 모드로 열기
        with open(wav_scp, mode='r') as file_wav, \
                open(feat_scp, mode='w') as file_feat:
            # wav 리스트를 한 줄씩 읽음
            for line in file_wav:
                # 각 행에는 발화 ID와 wav 파일 경로가
                # 스페이스로 구분되어 있으므로
                # split 함수를 써서 스페이스 구분 행을
                # 리스트형 변수로 변환
                parts = line.split()
                # 0번째가 발화 ID
                utterance_id = parts[0]
                # 1번째가 wav 파일 경로
                wav_path = parts[1]
                
                # wav 파일을 읽고 특징값 계산
                with wave.open(wav_path) as wav:
                    # 샘플링 주파수 확인
                    if wav.getframerate() != sample_frequency:
                        sys.stderr.write('The expected \
                            sampling rate is 16000.\n')
                        exit(1)
                    # wav 파일이 1채널(모노) 데이터인지 확인
                    # 데이터 여부를 확인
                    if wav.getnchannels() != 1:
                        sys.stderr.write('This program \
                            supports monaural wav file only.\n')
                        exit(1)
                    
                    # wav 데티어의 샘플 수
                    num_samples = wav.getnframes()

                    # wav 데이터를 읽어들임
                    waveform = wav.readframes(num_samples)

                    # 읽어온 데이터는 바이너리 값
                    # (16bit integer)이므로 숫자(정수)로 변환
                    waveform = np.frombuffer(waveform, dtype=np.int16)
                    
                    # MFCC 계산
                    mfcc = feat_extractor.ComputeMFCC(waveform)

                # 특징량의 프레임 수와 차원 수를 가져옴
                (num_frames, num_dims) = np.shape(mfcc)

                # 특징값 파일 이름(splitext로 확장자 제거)
                #out_file = os.path.splitext(os.path.basename(wav_path))[0]
                #out_file = os.path.join(os.path.abspath(out_dir), 
                #                        out_file + '.bin')

                out_file = os.path.join(os.path.abspath(out_dir), utterance_id + '.bin')
                # out_file = os.path.join(out_dir, utterance_id + '.bin')

                # 출력 경로에 필요한 디렉토리가 없으면 생성
                strbuf = os.path.dirname(out_file)
                if not os.path.exists(strbuf):
                    os.makedirs(strbuf)

                # 데이터를 float32 형식으로 변환
                mfcc = mfcc.astype(np.float32)

                # 데이터를 파일에 출력
                mfcc.tofile(out_file)
                # 발화ID, 특징 파일 경로, 프레임 수,
                # 차원 수를 특징 리스트에 기록
                file_feat.write("%s %s %d %d\n" %
                    (utterance_id, out_file, num_frames, num_dims))

