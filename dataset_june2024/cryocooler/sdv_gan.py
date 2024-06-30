import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()

real_data = pd.read_csv('original.csv')
metadata.detect_from_dataframe(real_data)

metadata.visualize(
    show_table_details='full',
    output_filepath='my_metadata.png'
)


# Step 1: Create the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)

# Step 2: Train the synthesizer
synthesizer.fit(real_data)

# Step 3: Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000)
synthetic_data.to_csv('augmented.csv', index=False)
