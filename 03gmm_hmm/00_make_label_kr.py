# -*- coding: utf-8 -*-

# 
# 라벨 파일의 음소를 ID로 변환하여 저장합니다.
# 또한, 음소와 ID의 대응을 기재한 리스트를 출력합니다.
# 

# os 모듈을 임포트
import os

def phone_to_int(label_str, 
                 label_int, 
                 phone_list,
                 insert_sil=False):
    ''' 
    음소 리스트를 사용하여 라벨 파일의
    음소를 숫자로 변환합니다.
    label_str: 문자열로 기재된 라벨 파일
    label_int: 문자열을 숫자로 변환한 쓰기 대상의
               라벨 파일
    phone_list: 음소 리스트
    insert_sil: True일 경우, 텍스트의 처음과 끝에
                공백을 삽입합니다.
    '''
    # 각 파일을 엽니다.
    with open(label_str, mode='r') as f_in,             open(label_int, mode='w') as f_out:
        # 라벨 파일을 한 줄씩 읽어옵니다.
        for line in f_in:
            # 읽은 줄을 공백으로 나누어
            # 리스트형 변수로 만듭니다.
            text = line.split()
            
            # 리스트의 0번째 요소는 발화 ID이므로,
            # 그대로 출력합니다.
            f_out.write('%s' % text[0])

            # insert_sil이 True이면,
            # 처음에 0(포즈)을 삽입합니다.
            if insert_sil:
                f_out.write(' 0')

            # 리스트의 1번째 이후 요소는 문자이므로,
            # 한 글자씩 숫자로 변환합니다.
            for u in text[1:]:
                # 음소 리스트에 없는 경우 오류
                if not u in phone_list:
                    sys.stderr.write('phone_to_int: \
                        unknown phone %s' % u)
                    exit(1)
                # 음소의 인덱스를 출력
                f_out.write(' %d' %                     (phone_list.index(u)))

            # insert_sil이 True이면,
            # 마지막에 0(포즈)을 삽입합니다.
            if insert_sil:
                f_out.write(' 0')
            # 줄바꿈
            f_out.write('\n')


# 
# 메인 함수
# 
if __name__ == "__main__":
    # 훈련 데이터의 라벨 파일 경로
    label_train_str =         '../data/label/train/text_phone'

    # 훈련 데이터의 처리 결과 출력 디렉토리
    out_train_dir =         './exp/data/train'

    # 개발 데이터의 라벨 파일 경로
    # (개발 데이터는 GMM-HMM에는 사용하지 않지만,
    # DNN-HMM에서 사용합니다.)
    #label_dev_str =         '../data/label/dev/text_phone'

    # 개발 데이터의 처리 결과 출력 디렉토리
    #out_dev_dir =         './exp/data/dev'

    # 음소 리스트
    phone_file = './cmu39.txt'

    # 포즈를 나타내는 기호
    silence_phone = 'pau'

    # True일 경우, 문장의 처음과 끝에 포즈를 삽입합니다.
    insert_sil = True

    # 음소 리스트의 처음에는 포즈 기호를 넣어 둡니다.
    phone_list = [silence_phone]
    # 음소 리스트 파일을 열어 phone_list에 저장
    with open(phone_file, mode='r') as f:
        for line in f:
            # 공백이나 줄바꿈을 제거하고 음소 기호를 가져옴
            phone = line.strip()
            # 음소 리스트의 끝에 추가
            phone_list.append(phone)


    # 훈련/개발 데이터 정보를 리스트화
    label_str_list = [label_train_str]
    out_dir_list = [out_train_dir]

    # 훈련/개발 데이터를 각각 처리
    for (label_str, out_dir)            in zip(label_str_list, out_dir_list):

        # 출력 디렉토리가 존재하지 않는 경우 생성
        os.makedirs(out_dir, exist_ok=True)

        # 음소와 숫자의 대응 리스트를 출력
        out_phone_list =             os.path.join(out_dir, 'phone_list')
        with open(out_phone_list, 'w') as f:
            for i, phone in enumerate(phone_list):
                # 리스트에 등록된 순서를
                # 그 음소에 대응하는 숫자로 함
                f.write('%s %d\n' % (phone, i))
     
        # 라벨의 음소 기호를 숫자로 변환하여 출력
        label_int =             os.path.join(out_dir, 'text_int')
        phone_to_int(label_str, 
                     label_int, 
                     phone_list, 
                     insert_sil)
