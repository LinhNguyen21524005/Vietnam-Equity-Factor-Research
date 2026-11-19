import vnstock
from vnstock import Market

print("Vnstock version:", vnstock.__version__)

market = Market()

print("Market initialized successfully.")

vnindex_test = market.index("VNINDEX").ohlcv(
    start="2020-01-01",
    end="2025-12-31"
)

print("Observations:", len(vnindex_test))
print("Columns:", vnindex_test.columns.tolist())

display(vnindex_test.head())
display(vnindex_test.tail())