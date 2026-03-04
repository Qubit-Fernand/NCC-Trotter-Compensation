def gen_q_tuples(j, s, q_min, q_max):
    """生成并迭代所有长度为 j 的整数元组 (q1,...,qj)，满足
    q_min <= qi <= q_max 且 sum(qi) == s。"""
    if j <= 0:
        if s == 0:
            yield ()
        return

    def helper(prefix, remaining_j, remaining_s):
        if remaining_j == 1:
            if q_min <= remaining_s <= q_max:
                yield tuple(prefix + [remaining_s])
            return
        # 给下一个值做剪枝：剩余和必须在可行范围内
        min_next = q_min
        max_next = q_max
        for val in range(min_next, max_next + 1):
            rem = remaining_s - val
            min_rem = q_min * (remaining_j - 1)
            max_rem = q_max * (remaining_j - 1)
            if rem < min_rem or rem > max_rem:
                continue
            yield from helper(prefix + [val], remaining_j - 1, rem)

    yield from helper([], j, s)

gen_q_tuples(2, s, q_min=K+1, q_max=q_0)