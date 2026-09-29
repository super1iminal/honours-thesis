# Honours Thesis

GPU-accelerated Monte Carlo simulation for verifying that autonomous controllers fail rarely, written for my UBC Computer Science honours thesis. The code fits minimum volume enclosing ellipses to millions of simulated states to partition the state space for Quasi-Markov Chain failure bounds, and explores eigenvalue-based partitions for a learned cart-pole controller. The full thesis is in [docs/Honours_Thesis.pdf](docs/Honours_Thesis.pdf).

## Layout

- `docs/` holds the thesis PDF.
- `toy_controller/` is the CUDA C region finder for a PD controller with Gaussian noise.
  - `src/` has the simulation, bisection search, gradient descent and region-finding code.
  - `plotting_script/` plots sampled states with their fitted ellipses.
  - `plot_info/` stores the states and ellipses written by a run.
  - `plots/` contains the rendered figures used in the thesis.
  - `logs/` collects time-stamped run logs, which git ignores.
  - `images/` keeps equation screenshots from early notes.
- `cart_pole/` covers the learned cart-pole controller.
  - `CUDA C/` is a CUDA C++ prototype that simulates the cart-pole with the learned policy and Lyapunov network on the GPU, along with a cubic spline kernel and its tests.
  - `python/` classifies states by the eigenvalues of their Jacobians and tracks transitions between modes, and includes the learned controller weights and a Conda environment.

## Running

Both CUDA programs need an NVIDIA GPU and `nvcc`, and their Makefiles target `sm_86`.

### Toy controller

From `toy_controller/`

```bash
make
./run -t 5
```

`./run -h` lists the options for sample count, time horizon, controller gains and log level. On the UBC GPU cluster, `sbatch run_program.sh` submits the same run as a job.

To plot a run, install pandas and matplotlib and then, from `toy_controller/plotting_script/`

```bash
python plot.py -t 5
```

### Cart-pole

The CUDA prototype in `cart_pole/CUDA C/` builds with `make`, and `make test` builds the spline tests as `run_tests`.

The Python analysis runs from `cart_pole/python/` in the included Conda environment.

```bash
conda env create -f environment.yml
conda activate ubc_thesis
python cp_eigdata.py
```

`cp_eigdata.py` and `cp_eigsummary.py` read the included `cp_jacobians.pkl`. `cp.py` loads `storage_unit.pkl`, which is too large for the repository, so on a fresh clone change the `data_processing(load=True)` call at the bottom of `cp.py` to `data_processing(save=True)` to generate it first.
