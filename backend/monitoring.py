from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from config.settings import MODEL_PRICING


@dataclass
class RequestLog:
    timestamp: str
    action: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool = True


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
    return round(
        (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000,
        8,
    )


class CostMonitor:
    def __init__(self):
        self.logs: List[RequestLog] = []

    def record(
        self,
        action: str,
        usage: dict,
        latency_ms: float,
        success: bool = True,
    ) -> RequestLog:
        model = usage.get("model", "unknown")
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)
        cost = calculate_cost(model, input_tok, output_tok)

        log = RequestLog(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            action=action,
            model=model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost,
            latency_ms=round(latency_ms, 0),
            success=success,
        )
        self.logs.append(log)
        return log

    @property
    def total_cost(self) -> float:
        return sum(l.cost_usd for l in self.logs)

    @property
    def total_tokens(self) -> int:
        return sum(l.input_tokens + l.output_tokens for l in self.logs)

    def as_records(self) -> list[dict]:
        return [
            {
                "Time": l.timestamp,
                "Action": l.action,
                "Model": l.model,
                "Input Tokens": l.input_tokens,
                "Output Tokens": l.output_tokens,
                "Cost (USD)": f"${l.cost_usd:.6f}",
                "Latency (ms)": l.latency_ms,
                "Status": "✅" if l.success else "❌",
            }
            for l in self.logs
        ]
