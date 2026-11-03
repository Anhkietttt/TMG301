"""Optional CLI: PYTHONPATH=src python scripts/run.py --config configs/pilot.json."""
import argparse
import json
from pathlib import Path
from domainmt.experiment import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/pilot.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    experiment = Experiment(json.loads(args.config.read_text()), args.output)
    print(experiment.prepare_data())
    print(experiment.generate_targets().head())
    print(experiment.backtranslate_and_rank())
    print(experiment.train_all())
    experiment.average_checkpoint()
    print(experiment.evaluate_all())
    print(experiment.bootstrap_improvement())


if __name__ == '__main__':
    main()
