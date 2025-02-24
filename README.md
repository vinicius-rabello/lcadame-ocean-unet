# Data Assimilation with Ensemble Kalman Filter (EnKF)

**brief explanation**

## Features
- Uses **numerical** and **ML-based** models for ensemble forecasts
- Implements **Ensemble Kalman Filter (EnKF)** for data assimilation
- Supports **observations with noise** and **perturbed initial conditions**
- Computes **error metrics (MSE, RMSE)** over the simulation period
- Generates **visualizations** for the updated state and error metrics

## Setup
### Clone the Repository
```sh
git clone https://github.com/vinicius-rabello/lcadame-ocean-unet.git
```

### Prerequisites
Make sure you have Python installed along with the necessary dependencies. You can install them using:
```sh
pip install -r requirements.txt
```

Or if you want to create a virtual environment:
```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Folder Structure
- `data/` → Contains input datasets (e.g., `oneYear.npy`)
- `utils/` → Includes helper functions for plotting and metric calculations
- `constants.py` → Defines simulation parameters
- `DA/` → Contains numerical and ML-based time-stepping functions
- `KalmanFilters/` → Implements EnKF for state update

## How It Works
### Initialization
1. **Load dataset** (`oneYear.npy`) containing the true state evolution
2. **Define observation noise** and extract observations at regular intervals
3. **Initialize ensemble members** using perturbed initial conditions

### Time Advancement Loop
- At each time step:
  - Advance **true state** and **ensemble predictions**
  - Every `DA_cycles` steps, apply **data assimilation** using EnKF
  - Compute **error metrics** (MSE, RMSE)
  - Save **updated state** and generate plots

### Final Metrics
- Compute mean MSE values over all time steps
- Display final error statistics

## Running the Code
Execute the main script to start the simulation:
```sh
python main.py
```

## Customization
- Modify `constants.py` to adjust parameters such as:
  - `DA_cycles`: Frequency of data assimilation steps
  - `sig_m`: Measurement noise standard deviation
  - `sig_b`: Initial condition noise standard deviation
  - `N, M`: Number of ML and numerical ensemble members
- Change `plotResults.py` to customize visualizations

## TODO
**i dont know**

## License
**i dont know**

## Author
**i dont know**