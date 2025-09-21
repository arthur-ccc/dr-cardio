from .mock import *

df = generate_mock_dataset(n_samples=1_000_000) 
export(df)