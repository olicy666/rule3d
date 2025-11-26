from __future__ import annotations

import argparse

from raven3d.dataset import DatasetGenerator, GenerationConfig
from raven3d.factory import create_default_registry
from raven3d.rules.base import RuleDifficulty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three-step 3D reasoning samples.")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated samples")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--points", type=int, default=4096, help="Number of points per point cloud")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--simple-prob", type=float, default=0.7, help="Probability for simple rules")
    parser.add_argument("--medium-prob", type=float, default=0.2, help="Probability for medium rules")
    parser.add_argument("--complex-prob", type=float, default=0.1, help="Probability for complex rules")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probs = {
        RuleDifficulty.SIMPLE: args.simple_prob,
        RuleDifficulty.MEDIUM: args.medium_prob,
        RuleDifficulty.COMPLEX: args.complex_prob,
    }
    config = GenerationConfig(n_points=args.points, difficulty_probs=probs)
    registry = create_default_registry()
    generator = DatasetGenerator(registry, config=config, seed=args.seed)
    generator.generate_dataset(args.output, args.num_samples)
    print(f"Generated {args.num_samples} samples in {args.output}")


if __name__ == "__main__":
    main()
