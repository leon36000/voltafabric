#!/usr/bin/env python3
"""Independent high-precision fixture oracle for R700's pinned gate route.

This script uses only the Python standard library. It computes the mathematical
fixture from scratch with Decimal rather than importing VoltaFabric/Rust or
DeepSeek donor code. It is a numerical oracle for one synthetic route, not a
checkpoint or PyTorch backend-parity test.
"""

from decimal import Decimal, getcontext
import json


EXPERTS = 384
TOP_K = 6
ROUTE_SCALE = Decimal("1.5")
EPSILON = Decimal("1e-20")
THRESHOLD = Decimal("20")


def softplus(value: Decimal) -> Decimal:
    if value > THRESHOLD:
        return value
    return (Decimal(1) + value.exp()).ln()


def main() -> None:
    getcontext().prec = 60
    rows = []
    for expert in range(EXPERTS):
        logit = Decimal(expert)
        score = softplus(logit).sqrt()
        correction = Decimal(1000) if expert == 0 else Decimal(0)
        rows.append((score + correction, expert, score))

    rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = rows[:TOP_K]
    denominator = sum((row[2] for row in selected), Decimal(0)) + EPSILON
    result = {
        "expert_ids": [row[1] for row in selected],
        "coefficients": [str(row[2] / denominator * ROUTE_SCALE) for row in selected],
        "coefficient_sum": str(
            sum((row[2] / denominator * ROUTE_SCALE for row in selected), Decimal(0))
        ),
        "precision_digits": getcontext().prec,
        "fixture": "logit[i]=i, text_bias[0]=1000, remaining biases=0",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
