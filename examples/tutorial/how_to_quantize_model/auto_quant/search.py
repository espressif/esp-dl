"""Strategy search.

Two-level expansion:
    1. strategy_space: which modules are ON/OFF in this strategy
    2. param_space:    for each ON module, which knob values to try

Both are cartesian-producted. Use rules to prune impossible combos.
"""

import itertools


class StrategyCompiler:
    """Expand strategy_space + param_space. No filtering, no rules."""

    def __init__(self, strategy_space: dict, param_space: dict):
        self.strategy_space = strategy_space
        self.param_space = param_space

    def _expand_param(self, name: str):
        """Expand a module's knob space into a list of concrete dicts.

        Rules per knob value:
            - list  : discrete enumeration
            - tuple : (start, end, step) grid sampling, rounded to 1 decimal
        """
        if name not in self.param_space:
            # Module is enabled but has no knobs -> single empty config.
            return [{}]

        raw = self.param_space[name]
        keys = list(raw.keys())
        dim_values = []

        for k in keys:
            v = raw[k]
            if isinstance(v, list):
                dim_values.append(v)
            elif isinstance(v, tuple):
                start, end, step = v
                samples = []
                idx = 0
                while True:
                    value = start + idx * step
                    if value > end + 1e-12:
                        break
                    samples.append(round(value, 1))
                    idx += 1
                dim_values.append(samples)
            else:
                raise TypeError(
                    f"Unsupported param format: {name}.{k}={v!r}; "
                    "must be list or (start, end, step)"
                )

        return [dict(zip(keys, combo)) for combo in itertools.product(*dim_values)]

    def build(self):
        """Return fully expanded strategy list."""
        expanded = []
        keys = list(self.strategy_space.keys())
        values = list(self.strategy_space.values())

        for combo in itertools.product(*values):
            base = dict(zip(keys, combo))
            strategy = {}
            for k, v in base.items():
                if v is False:
                    strategy[k] = {"value": False, "param_candidates": None}
                else:
                    strategy[k] = {
                        "value": v,
                        "param_candidates": self._expand_param(k),
                    }
            expanded.append(strategy)
        return expanded


class StrategyRules:
    """Constraint engine. Add `lambda strategy: bool` rules; all must pass."""

    def __init__(self):
        self.rules = []

    def add(self, rule_fn):
        self.rules.append(rule_fn)

    def valid(self, strategy) -> bool:
        return all(rule(strategy) for rule in self.rules)


class ExhaustiveStrategySearcher:
    """Yield strategies that satisfy every rule."""

    def __init__(self, strategies, rules: StrategyRules = None):
        self.strategies = strategies
        self.rules = rules or StrategyRules()

    def search(self):
        for strategy in self.strategies:
            if self.rules.valid(strategy):
                yield strategy


def build_search_pipeline(strategy_space, param_space):
    compiler = StrategyCompiler(strategy_space, param_space)
    rules = StrategyRules()
    searcher = ExhaustiveStrategySearcher(compiler.build(), rules)
    return searcher, rules
