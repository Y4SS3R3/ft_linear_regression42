from utils import GREEN, RESET
from utils import fail, load_dataset, load_thetas, normalize, denormalize, estimatePrice

t0, t1, min_km, max_km, min_price, max_price = load_thetas()

df = load_dataset()

cars = len(df)

if max_km - min_km == 0:
    fail("every car has the same mileage, the model is undefined.")

km_normalized = normalize(df['km'], min_km, max_km)

predicted_prices = denormalize(estimatePrice(km_normalized, t0, t1), min_price, max_price)

error = df['price'] - predicted_prices

ss_res = (error ** 2).sum(skipna=False)

mae = error.abs().mean() # remove the direction of the error

mean_price = df['price'].mean()
ss_tot = ((df['price'] - mean_price) ** 2).sum(skipna=False) # to avoid skipping NaN when the learning rate = 2, since sum skips NaN the ss_tot would be 0

if ss_tot == 0:
    fail("every car has the same price, R² is undefined.")

r_squared = 1 - (ss_res / ss_tot) # if ss_res was 0, we would would get 100% coverage, which is a misleading incorrect result

print(f"{GREEN}➜ Cars evaluated : {cars}{RESET}")
print(f"{GREEN}➜ R² (precision) : {r_squared:.4f} ({r_squared * 100:.2f}%){RESET}")
print(f"{GREEN}➜ MAE            : {mae:.2f} {RESET}")
