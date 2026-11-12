"""Flower 1.42 app (message API: ClientApp / ServerApp), DECISIONS F1-F8.

Training lives in a pluggable `train_fn`, so the same wiring runs the step-A toy now and Fidel's client_update
later. The server strategy is always Flower's plain FedAvg: FedProx's proximal term and FedLA's sample weights
live in the client loss (F4, LA5), so all three arms share one server.

    train_fn(arrays, client_id, config) -> (new_arrays, n_training_windows)
    evaluate_fn(server_round, arrays) -> {"metric": value, ...}     # server-side, all validation engines (F6)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation

Arrays = list[np.ndarray]
TrainFn = Callable[[Arrays, int, dict], tuple[Arrays, int]]
EvalFn = Callable[[int, Arrays], dict[str, float]]


@dataclass
class FlowerResult:
    history: list[dict] = field(default_factory=list)   # one entry per round, round 0 = initial weights
    final: Arrays | None = None                         # weights after the last round
    best_round: int | None = None                       # round with the lowest best_key (F6)
    best_arrays: Arrays | None = None


def make_strategy(K: int) -> FedAvg:
    """Plain FedAvg, every client every round (F1), weighted by window count (F3), no client-side evaluation."""
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    return FedAvg(fraction_train=1.0, fraction_evaluate=0.0,
                  min_train_nodes=K, min_available_nodes=K, weighted_by_key="num-examples")


def make_client_app(train_fn: TrainFn) -> ClientApp:
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        # Stateless (F2): nothing is read from or written to context.state between rounds.
        arrays = msg.content["arrays"].to_numpy_ndarrays()
        config = dict(msg.content["config"]) if "config" in msg.content else {}
        client_id = int(context.node_config["partition-id"])
        new_arrays, n = train_fn(arrays, client_id, config)
        return Message(content=RecordDict({"arrays": ArrayRecord(list(new_arrays)),
                                           "metrics": MetricRecord({"num-examples": int(n)})}),
                       reply_to=msg)

    return app


def make_server_app(initial: Arrays, K: int, num_rounds: int, train_config: dict | None,
                    evaluate_fn: EvalFn | None, best_key: str | None, out: FlowerResult) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        def server_eval(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            if evaluate_fn is None:
                return None
            w = arrays.to_numpy_ndarrays()
            metrics = {k: float(v) for k, v in evaluate_fn(server_round, w).items()}
            out.history.append({"round": server_round, "metrics": metrics})
            # Lowest wins; ties keep the earlier round.
            if best_key is not None and (out.best_round is None
                                         or metrics[best_key] < out.history[out.best_round]["metrics"][best_key]):
                out.best_round, out.best_arrays = server_round, [a.copy() for a in w]
            return MetricRecord(metrics)

        result = make_strategy(K).start(grid=grid, initial_arrays=ArrayRecord(list(initial)),
                                        num_rounds=num_rounds, train_config=ConfigRecord(train_config or {}),
                                        evaluate_fn=server_eval)
        out.final = result.arrays.to_numpy_ndarrays()

    return app


def run_flower(train_fn: TrainFn, initial: Arrays, K: int, num_rounds: int, train_config: dict | None = None,
               evaluate_fn: EvalFn | None = None, best_key: str | None = None,
               cpus_per_client: int = 1) -> FlowerResult:
    """Run one federated training in Flower's simulation engine (Ray backend) and return its history."""
    out = FlowerResult()
    run_simulation(server_app=make_server_app(initial, K, num_rounds, train_config, evaluate_fn, best_key, out),
                   client_app=make_client_app(train_fn), num_supernodes=K,
                   backend_config={"client_resources": {"num_cpus": cpus_per_client}})
    if out.final is None:
        raise RuntimeError("Flower simulation ended without a final model")
    return out
