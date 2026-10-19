def edit_distance(target_text: str, pred_text: str):
    dp = [[0 for j in range(len(pred_text) + 1)] for i in range(len(target_text) + 1)]

    for i in range(1, len(target_text) + 1):
        dp[i][0] = dp[i - 1][0] + 1

    for j in range(1, len(pred_text) + 1):
        dp[0][j] = dp[0][j - 1] + 1

    for i in range(1, len(target_text) + 1):
        for j in range(1, len(pred_text) + 1):
            dp[i][j] = min(
                dp[i][j - 1] + 1,
                dp[i - 1][j] + 1,
                dp[i - 1][j - 1] + int(target_text[i - 1] != pred_text[j - 1]),
            )

    return dp[-1][-1]


def calc_wer(target_text: str, pred_text: str) -> float:
    return edit_distance(
        target_text=target_text.split(), pred_text=pred_text.split()
    ) / max(1, len(target_text.split()))


def calc_cer(target_text: str, pred_text: str) -> float:
    return edit_distance(target_text=target_text, pred_text=pred_text) / max(
        1, len(target_text)
    )
