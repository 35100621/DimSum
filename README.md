# DimSum

DimSum trains a PPO policy to compose bounded image transformations against a
DeepfakeBench Xception victim model. The detector is isolated behind
`victim.detector.VictimDetector`; PPO and evaluation code receive only normalized
REAL/FAKE probabilities.

The intended workflow is documented as five runnable notebooks:

1. `notebooks/01_detector_setup.ipynb` configures and checks Xception.
2. `notebooks/02_attack_baselines.ipynb` visualizes every bounded action.
3. `notebooks/03_ppo_environment.ipynb` validates reward and episode behavior.
4. `notebooks/04_ppo_training.ipynb` trains and saves PPO.
5. `notebooks/05_evaluation.ipynb` evaluates a separate held-out directory.

Create an environment and install dependencies with `pip install -r requirements.txt`.
Edit only the path cell near the top of notebooks 01, 04, and 05. Training and test
directories must remain separate.

The same workflow is available from the command line:

```bash
python -m ppo.train --help
python -m evaluation.evaluate_attack --help
pytest -q
```
