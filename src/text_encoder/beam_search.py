from numpy import logaddexp


def ctc_beam_search(log_probs, blank_id=0, beam_size=10):
    beam = {(): [0.0, float("-inf")]}
    for log_prob_t in log_probs.detach().float().cpu().tolist():
        new_beam = {}
        for prefix, (p_blank, p_nonblank) in beam.items():
            total = logaddexp(p_blank, p_nonblank)
            for token, prob in enumerate(log_prob_t):
                if prob == float("-inf"):
                    continue
                if prefix not in new_beam:
                    new_beam[prefix] = [float("-inf"), float("-inf")]
                if token != blank_id and prefix + (token,) not in new_beam:
                    new_beam[prefix + (token,)] = [float("-inf"), float("-inf")]
                if token == blank_id:
                    new_beam[prefix][0] = logaddexp(new_beam[prefix][0], total + prob)
                elif prefix and prefix[-1] == token:
                    new_beam[prefix][1] = logaddexp(
                        new_beam[prefix][1], p_nonblank + prob
                    )
                    new_beam[prefix + (token,)][1] = logaddexp(
                        new_beam[prefix + (token,)][1], p_blank + prob
                    )
                else:
                    new_beam[prefix + (token,)][1] = logaddexp(
                        new_beam[prefix + (token,)][1], total + prob
                    )
        beam = dict(
            sorted(
                new_beam.items(),
                key=lambda x: logaddexp(*x[1]),
                reverse=True,
            )[:beam_size]
        )

    return [(prefix, logaddexp(*log_probs)) for prefix, log_probs in beam.items()]
