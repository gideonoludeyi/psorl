import argparse

from .experiment import experiment


def main():
    parser = argparse.ArgumentParser(description="Run the TERL experiment.")
    parser.add_argument(
        "--num_agents", type=int, default=25, help="Number of agents in the population"
    )
    parser.add_argument(
        "--replay_buffer_capacity",
        type=int,
        default=10_000,
        help="Capacity of the replay buffer",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=100_000,
        help="Maximum number of timesteps for the experiment",
    )
    parser.add_argument(
        "--exploration_ratio",
        type=float,
        default=0.25,
        help="Ratio of timesteps for exploration phase",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for agent updates",
    )
    parser.add_argument(
        "--seed", type=int, required=False, help="Random seed for reproducibility"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    experiment(
        num_agents=args.num_agents,
        replay_buffer_capacity=args.replay_buffer_capacity,
        max_timesteps=args.max_timesteps,
        exploration_ratio=args.exploration_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
