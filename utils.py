import sys

import json

import pandas as pd

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def fail(message):
    print(f"{RED}✗ Error: {message}{RESET}")
    sys.exit(1)


def load_dataset(path="data.csv"):
    try:
        df = pd.read_csv(path).dropna().reset_index(drop=True)
    except FileNotFoundError:
        fail(f"The file {path} was not found.")
    except Exception as e:
        fail(f"{e}.")

    if len(df) == 0:
        fail(f"{path} holds no usable rows.")

    return df


def load_thetas(path="thetas.json"):
    try:
        with open(path) as file:
            data = json.load(file)
            return (data['t0'], data['t1'],
                    data['min_km'], data['max_km'],
                    data['min_price'], data['max_price'])
    except json.decoder.JSONDecodeError:
        fail(f"The file {path} is empty or corrupted.")
    except FileNotFoundError:
        # not trained yet: fall back to a flat model, every price is 0
        print(f"{YELLOW}➜ Warning: {path} was not found, "
              f"using theta0 = 0 and theta1 = 0. Run train.py first.{RESET}")
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    except KeyError as e:
        fail(f"{path} is missing the required key: {e}")
    except Exception as e:
        fail(f"{e}.")


def normalize(value, minimum, maximum):
    if maximum - minimum == 0:
        return 0.0 # avoid ZeroDivisionError
    return (value - minimum) / (maximum - minimum)


def denormalize(value, minimum, maximum):
    return (value * (maximum - minimum)) + minimum


def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage) # Series.__rmul__(float)
