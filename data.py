import numpy as np

def read_file(filename):
    """
    Reads a CSV file and returns the data as a NumPy array.

    Each row in the CSV should have 12 numbers:
    the first 6 are avgWire values and the last 6 are slope values.

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    np.ndarray, shape [N, 12]
        Array containing all events from the file.

    Raises
    ------
    ValueError
        If any row does not contain exactly 12 numbers.
    """
    events = []
    with open(filename, 'r') as f:
        # Read non-empty lines
        lines = [line.strip() for line in f if line.strip() != ""]

    for line in lines:
        values = [float(x) for x in line.split(",")]
        if len(values) != 12:
            raise ValueError(f"Each row must have 12 numbers, but got {len(values)}: {line}")
        events.append(values)

    return np.array(events, dtype=np.float32)  # [N, 12]
