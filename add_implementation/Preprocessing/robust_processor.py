# add_implementation/Preprocessing/robust_processor.py

import numpy as np


class RobustProcessor:
    """
    간단한 에너지 기반 VAD + 발화 단위 CMVN을 수행하는 전처리기.

    - 입력: (num_frames, num_dims) 형태의 MFCC 특징 (예: 39차원)
    - 출력: VAD로 무음 프레임 제거 후, CMVN이 적용된 특징
    """

    def __init__(
        self,
        use_vad: bool = True,
        use_cmvn: bool = True,
        min_speech_frames: int = 5,
        hangover: int = 3,
        energy_percentile_speech: float = 70.0,
        energy_percentile_noise: float = 10.0,
        eps: float = 1e-8,
    ):
        """
        Args:
            use_vad: VAD 사용 여부
            use_cmvn: CMVN 사용 여부
            min_speech_frames: 이 값보다 짧은 연속 구간은 음성으로 보지 않고 버림
            hangover: 한 프레임을 음성으로 판단했을 때, 앞뒤로 같이 음성으로 묶어주는 프레임 수
            energy_percentile_speech: 에너지 상위 몇 %를 'speech 후보'로 볼지
            energy_percentile_noise: 에너지 하위 몇 %를 'noise 후보'로 볼지
            eps: 분산 0 방지를 위한 작은 값
        """
        self.use_vad = use_vad
        self.use_cmvn = use_cmvn
        self.min_speech_frames = min_speech_frames
        self.hangover = hangover
        self.energy_percentile_speech = energy_percentile_speech
        self.energy_percentile_noise = energy_percentile_noise
        self.eps = eps

    # -----------------------------
    # 공개 인터페이스
    # -----------------------------
    def process(self, feats: np.ndarray) -> np.ndarray:
        """
        전체 전처리 파이프라인.

        Args:
            feats: (T, D) MFCC 특징

        Returns:
            (T', D) 처리된 특징
        """
        if feats is None:
            return feats
        if feats.ndim != 2:
            raise ValueError(f"feats must be 2-D (T, D), got shape {feats.shape}")

        processed = feats

        if self.use_vad:
            processed = self._apply_vad(processed)
            # 전부 잘려나가면 원본 그대로 쓰도록 fallback
            if processed.size == 0:
                processed = feats

        if self.use_cmvn:
            processed = self._apply_cmvn(processed)

        return processed

    # -----------------------------
    # VAD
    # -----------------------------
    def _frame_energy(self, feats: np.ndarray) -> np.ndarray:
        """
        프레임별 에너지 추정.

        - 39차 MFCC에서 0번 계수(c0)를 에너지 proxy로 쓰는 게 일반적이지만,
          혹시 구조가 다를 수 있으므로 안전하게 '프레임 벡터의 L2-norm'을 사용.
        """
        # (T,)
        energy = np.linalg.norm(feats, axis=1)
        return energy

    def _apply_vad(self, feats: np.ndarray) -> np.ndarray:
        """
        에너지 기반 voice activity detection.

        1) 프레임 에너지 분포에서 하위 p% = noise, 상위 q% = speech 후보
        2) 두 값 사이에 threshold 설정
        3) threshold 이상인 프레임을 음성으로 판단
        4) hangover와 min_speech_frames로 다듬기
        """
        num_frames = feats.shape[0]
        if num_frames == 0:
            return feats

        energy = self._frame_energy(feats)

        # 에너지 통계로 threshold 설정
        noise_level = np.percentile(energy, self.energy_percentile_noise)
        speech_level = np.percentile(energy, self.energy_percentile_speech)

        # noise < threshold < speech 가 되도록 중간 값 사용
        thresh = (noise_level + speech_level) / 2.0

        # 1차 VAD 마스크
        vad_mask = energy > thresh

        # hangover: 한 프레임이 speech이면 앞뒤 몇 개도 speech로 확장
        if self.hangover > 0:
            vad_mask = self._apply_hangover(vad_mask, self.hangover)

        # 너무 짧은 구간 제거
        vad_mask = self._remove_short_segments(vad_mask, self.min_speech_frames)

        # 최종 speech 프레임만 남김
        return feats[vad_mask, :]

    @staticmethod
    def _apply_hangover(mask: np.ndarray, hangover: int) -> np.ndarray:
        """
        speech frame 주변으로 hangover 만큼 확장.
        """
        T = len(mask)
        new_mask = mask.copy()

        speech_indices = np.where(mask)[0]
        for idx in speech_indices:
            start = max(0, idx - hangover)
            end = min(T, idx + hangover + 1)
            new_mask[start:end] = True

        return new_mask

    @staticmethod
    def _remove_short_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
        """
        True가 연속된 구간이 min_len 미만이면 False로 바꾸는 후처리.
        """
        if min_len <= 1:
            return mask

        T = len(mask)
        new_mask = mask.copy()

        start = None
        for i in range(T + 1):
            if i < T and mask[i] and start is None:
                start = i  # 구간 시작
            elif (i == T or not mask[i]) and start is not None:
                # 구간 끝
                end = i
                length = end - start
                if length < min_len:
                    new_mask[start:end] = False
                start = None

        return new_mask

    # -----------------------------
    # CMVN
    # -----------------------------
    def _apply_cmvn(self, feats: np.ndarray) -> np.ndarray:
        """
        발화 단위 Cepstral Mean & Variance Normalization.

        x_norm = (x - mean) / std
        """
        if feats.size == 0:
            return feats

        mean = np.mean(feats, axis=0, keepdims=True)
        std = np.std(feats, axis=0, keepdims=True)

        std = np.where(std < self.eps, 1.0, std)

        return (feats - mean) / std