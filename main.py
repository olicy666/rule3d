from __future__ import annotations

import argparse

from raven3d.dataset import DatasetGenerator, GenerationConfig
from raven3d.factory import create_default_registry
from raven3d.rules.groups import list_available_modes, rules_for_mode, validate_rule_ids
from raven3d.rules.base import RuleDifficulty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three-step 3D reasoning samples.")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated samples")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--points", type=int, default=4096, help="Number of points per point cloud")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--simple-prob", type=float, default=0.7, help="Deprecated (rule sampling is uniform)")
    parser.add_argument("--medium-prob", type=float, default=0.2, help="Deprecated (rule sampling is uniform)")
    parser.add_argument("--complex-prob", type=float, default=0.1, help="Deprecated (rule sampling is uniform)")
    parser.add_argument(
        "--mode",
        type=str.lower,
        default="main",
        choices=list_available_modes(),
        help="Rule preset: main / r1-only / r2-only / r3-only / all-minus-r1 / all-minus-r2 / all-minus-r3",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default=None,
        help="Comma-separated rule IDs (e.g., R1-1,R2-3,R3-2). When provided, overrides --mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probs = {
        RuleDifficulty.SIMPLE: args.simple_prob,
        RuleDifficulty.MEDIUM: args.medium_prob,
        RuleDifficulty.COMPLEX: args.complex_prob,
    }
    try:
        rule_filter = validate_rule_ids(args.rules.split(",")) if args.rules else rules_for_mode(args.mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = GenerationConfig(n_points=args.points, difficulty_probs=probs, rule_filter=rule_filter)
    registry = create_default_registry()
    generator = DatasetGenerator(registry, config=config, seed=args.seed)
    generator.generate_dataset(args.output, args.num_samples)
    mode_info = f"custom rules [{', '.join(sorted(rule_filter))}]" if args.rules else f"mode '{args.mode}'"
    print(f"Generated {args.num_samples} samples in {args.output} with {mode_info}")


if __name__ == "__main__":
    main()
