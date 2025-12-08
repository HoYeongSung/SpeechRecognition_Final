# summarize_train_log.py
# 학습 로그(train_log.txt)에서 iteration별 log-likelihood 평균을 뽑아서
# train_log_summary.csv 파일에 "추가"하는 스크립트
# + 각 모델의 총 학습 시간(sec, min)도 같이 기록

import re
import os
from statistics import mean


# LOG_FILE = "train_log.txt"          # 로그 파일 이름
# OUT_FILE = "train_log_summary.csv"  # SGMM, GMM 둘 다 누적
#
# # 여기만 바꿔가면서 사용
# #   SGMM 요약할 때: MODEL_NAME = "sgmhmm"
# #   GMM 요약할 때:  MODEL_NAME = "gmmhmm"
# # MODEL_NAME = "sgmhmm"
# MODEL_NAME = "sgmhmm"

LOG_FILE  = "train_log_gmmhmm.txt"
OUT_FILE  = "train_log_summary.csv"
MODEL_NAME = "gmmhmm"




# 예: "1-th iterateion"
re_iter = re.compile(r"(\d+)-th iterateion")

# 예: "log likelihood averaged over 10 utterances: -5235.886091"
re_ll = re.compile(r"log likelihood averaged over 10 utterances:\s+(-?\d+\.\d+)")

# 예: "Total training time: 123.45 seconds (2.06 minutes)"
re_time = re.compile(
    r"Total training time:\s+([0-9.]+)\s+seconds\s+\(([0-9.]+)\s+minutes\)"
)

iter_ll = {}       # {iteration: [ll1, ll2, ...]}
current_iter = None

total_time_sec = None
total_time_min = None
global_iter = 0
# PowerShell 리다이렉트/ Tee-Object 때문에 UTF-16일 가능성 높아서 utf-16으로 읽기
with open(LOG_FILE, "r", encoding="utf-16") as f:
    for line in f:
        line = line.strip()

        m_iter = re_iter.search(line)
        if m_iter:
            global_iter += 1             # 1,2,...,20 이런 식으로 증가
            current_iter = global_iter   # 화면의 '1-th' 숫자는 무시하고
            iter_ll[current_iter] = []
            continue

        # log-likelihood 값 찾기
        m_ll = re_ll.search(line)
        if m_ll and current_iter is not None:
            ll_value = float(m_ll.group(1))
            iter_ll[current_iter].append(ll_value)
            continue

        # 총 학습 시간 찾기
        m_time = re_time.search(line)
        if m_time:
            total_time_sec = float(m_time.group(1))
            total_time_min = float(m_time.group(2))
            continue

# CSV가 이미 있는지 확인 → 없으면 헤더 한 번만 작성
write_header = not os.path.exists(OUT_FILE)

with open(OUT_FILE, "a", encoding="utf-16") as f:
    if write_header:
        # model, iteration별 log-likelihood + 총 학습 시간 컬럼
        f.write(
            "model,iteration,num_chunks,mean_log_likelihood,"
            "min_log_likelihood,max_log_likelihood,"
            "total_time_sec,total_time_min\n"
        )

    for it in sorted(iter_ll.keys()):
        values = iter_ll[it]
        if not values:
            continue
        avg_ll = mean(values)
        min_ll = min(values)
        max_ll = max(values)

        # total_time_* 이 None이면 빈 칸으로
        t_sec = f"{total_time_sec:.2f}" if total_time_sec is not None else ""
        t_min = f"{total_time_min:.2f}" if total_time_min is not None else ""

        f.write(
            f"{MODEL_NAME},{it},{len(values)},"
            f"{avg_ll},{min_ll},{max_ll},"
            f"{t_sec},{t_min}\n"
        )

print(f"Appended summary for model='{MODEL_NAME}' to {OUT_FILE}")
