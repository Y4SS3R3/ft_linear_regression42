import sys

from utils import RED, GREEN, YELLOW, RESET
from utils import load_thetas, normalize, denormalize, estimatePrice

t0, t1, min_km, max_km, min_price, max_price = load_thetas()

while True:

    try:
        user_in = input("➜ Insert car mileage: ")
        mileage_in = int(user_in)
        break
    except (KeyboardInterrupt, EOFError):
        print(f"\n{YELLOW}➜ Exit{RESET}")
        sys.exit(0)
    except ValueError:
        print(f"{RED}✗ Error: That wasn't a number try again.{RESET}")

# input must be normalized since we trained on normalized dataset
norm_mileage = normalize(mileage_in, min_km, max_km)

norm_price = estimatePrice(norm_mileage, t0, t1)

price = denormalize(norm_price, min_price, max_price)

print(f"{GREEN}➜ Car's price is: {max(price, 0):.2f}{RESET}")
