import argparse

from .experiment import experiment
from .td3_experiment import td3_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the TERL or TD3 experiment.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="Ant-v5",
        help="Continous control environment to use for the experiment [default: Ant-v5]",
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="terl",
        choices=["terl", "td3"],
        help="Type of experiment to run (terl or td3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--num_agents",
        type=int,
        default=25,
        help="[TERL] Number of agents in the population",
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
        "--discount", type=float, default=0.99, help="Discount factor for TD3"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="Soft update coefficient for TD3"
    )
    parser.add_argument(
        "--policy_noise",
        type=float,
        default=0.2,
        help="Std of noise added to target policy's action",
    )
    parser.add_argument(
        "--noise_clip",
        type=float,
        default=0.5,
        help="Range to clip target policy noise",
    )
    parser.add_argument(
        "--policy_freq",
        type=int,
        default=2,
        help="Frequency of delayed policy updates",
    )

    args = parser.parse_args()

    if args.experiment_type == "terl":
        experiment(
            env_name=args.env_name,
            seed=args.seed,
            num_agents=args.num_agents,
            replay_buffer_capacity=args.replay_buffer_capacity,
            max_timesteps=args.max_timesteps,
            exploration_ratio=args.exploration_ratio,
            batch_size=args.batch_size,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            verbose=args.verbose,
        )
    elif args.experiment_type == "td3":
        td3_experiment(
            env_name=args.env_name,
            seed=args.seed,
            max_timesteps=args.max_timesteps,
            replay_buffer_capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
