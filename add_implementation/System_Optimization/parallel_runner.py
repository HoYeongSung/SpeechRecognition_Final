# 멀티코어용 간단 map 함수
# num_workers=1 이면 그냥 순차 실행, 2 이상이면 multiprocessing.Pool 사용

import multiprocessing as mp
from typing import Iterable, Callable, Any, List, Optional


def parallel_map(
    func: Callable[[Any], Any],
    iterable: Iterable[Any],
    num_workers: Optional[int] = None,
) -> List[Any]:
    items = list(iterable)

    # 처리할게 없으면 바로 리턴
    if not items:
        return []

    # 워커가 1 이하면 순차 실행
    if num_workers is None or num_workers <= 1:
        return [func(x) for x in items]

    # 실제 코어 수보다 많이 쓰지 않도록 제한
    n_workers = min(num_workers, mp.cpu_count())

    # Pool.map 으로 병렬 실행 (입력 순서 유지됨)
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(func, items)

    return results
