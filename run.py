"""Entry point for the FFBA pipeline. Algorithm settings live in config.yaml."""

from ffba.config import parse_args


def main():
    args = parse_args()
    from ffba.pipeline import run_pipeline

    run_pipeline(args)


if __name__ == "__main__":
    main()
