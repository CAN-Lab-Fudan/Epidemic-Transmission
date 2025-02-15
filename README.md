# ViralDynamic
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamics/blob/master/logo.png" width="180px">

![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg) ![Python Version](https://img.shields.io/badge/python-%3E%3D%203.7-blue.svg) ![Matlab Version](https://img.shields.io/badge/Matlab%202018b-blue.svg)
 ![License](https://img.shields.io/badge/license-Apache%202.0-green.svg) ![Platform](https://img.shields.io/badge/platform-win%20%7C%20macos%20%7C%20linux-yellow.svg)


## 1. Introduction
**ViralDynamic** is a Python & Matlab framework specifically developed for the simulation and analysis of epidemic spreading on complex networks. The framework combines several essential functionalities, including temporal network construction, trajectory planning, higher-order dynamics prediction, propagation dynamics simulation, and immunization game evolution analysis. This tutorial aims to help you get started. By completing this tutorial, you will discover the core functionalities of **ViralDynamic**. 

Key features of **ViralDynamic** include:
1. **TeNet**: A temporal network modeling tool for large-scale WiFi data. Allows users to create dynamic networks that represent interactions within a population, simulating disease spread across different types of connections.

2. **SubTrack**: A trajectory planning tool for massive metro swipe card data. Supports users in obtaining precise individual trajectories during metro trips to analyze mobility patterns.

3. **TaHiP**: A high-order dynamics prediction tool. Allows users to accurately predict higher-order dynamic with unknown higher-order topology.

4. **ε-SIS**: A tool for simulating ε-susceptible-infected-susceptible (ε-SIS) dynamics in continuous time.

5. **EvoVax**: A social perception-based immunization game evolution analysis tool. 

<img src="https://github.com/CAN-Lab-Fudan/ViralDynamic/blob/master/ViralDynamic_framework.png" width="950px">

## 2. TeNet
### 2.1 Background Information

In dynamic population analysis and social network research, user contact data based on wireless access points (APs) is a valuable resource. This data could be used to analyze the frequency and duration of user contacts, as well as their temporal and spatial distributions, providing insights into mobility patterns, group behaviors, and social network modeling.
**TeNet** is a MATLAB-based framework for analyzing and processing user contact data. It enables data cleaning, construction of contact records, and statistical analysis of user interactions, ultimately generating contact networks and statistical summaries.
Using this tool, we processed the **FudanWiFi13** dataset, generating several key outputs that can assist researchers in analyzing and understanding user interactions.

Our paper is available at [this link](https://doi.org/10.1016/j.physa.2019.04.068). Citations are welcome.

```bibtex
@article{LI20191276,
title = {Can multiple social ties help improve human location prediction?},
journal = {Physica A: Statistical Mechanics and its Applications},
volume = {525},
pages = {1276-1288},
year = {2019},
issn = {0378-4371},
doi = {https://doi.org/10.1016/j.physa.2019.04.068},
url = {https://www.sciencedirect.com/science/article/pii/S0378437119304169},
author = {Cong Li and Shumin Zhang and Xiang Li},
keywords = {Social system, Location prediction, Familiar stranger, Social tie}
}
```

### 2.2 Key Features

**TeNet** includes the following core functionalities:

#### 1) Data Import and Preprocessing:
* `openDataSetMain.m`:
  * Import raw data, clean it, and perform unit conversion (e.g., convert contact duration from seconds to minutes).
  * Output formatted data files with userId, startTime, duration, and AP.
* `RecordMain.m`:
  * Filter records based on APs and time ranges.
  * Assign unique user IDs and output the processed user data.

#### 2) Contact Record Construction:
* `PairwiseInteractionNetworkCreator.m`:
  * Construct user contact records based on filtered user activity data.
  * Output user pairs, interaction durations, and timestamps.

#### 3) Contact Duration and Frequency Statistics:
* `StaticIntervalsCreator.m`:
  * Analyze contact records to calculate interaction durations and frequencies.
  * Output aggregated contact data.

#### 4) Contact Information Filtering:
* `PartlyAggregated.m`:
  * Filter specific time ranges or user groups from the contact records.
  * Output filtered contact files.

### 2.3 How to Run
#### 1) Prerequisites:
* MATLAB R2012b or later.
* Input data files in the following format:
  * **Raw data**: e.g., `records.mat` containing `userId`, `startTime`, `duration`, and `location`.
  * **API mapping file**: e.g., `tbaps.txt` mapping access points to users.

#### 2) Setup
1. Clone or download this repository:
```bash
git clone https://github.com/CAN-Lab-Fudan/ViralDynamic/tree/master/TeNet.git
cd contact-network-analysis
```
2. Place the raw data files (`records.mat` and `tbaps.txt`) in the root directory of the project.
3. Open MATLAB and set the current directory to the project folder.

#### 3) Running the Scripts:
Follow these steps to run the tool:

1. **Data Import and Preprocessing**:

   Run `openDataSetMain.m` and `RecordMain.m`:
   ```matlab
   openDataSetMain; % Data cleaning and unit conversion
   RecordMain; % Filtering and user ID assignment
   ```
   Output:
   * `tbdata.mat`: Preprocessed user data.

2. **Construct Contact Records**:

   Run `PairwiseInteractionNetworkCreator.m`:
   ```matlab
   PairwiseInteractionNetworkCreator;
   ```
   Output:
   * `tbPairwiseInteraction.txt`: User contact information.
   * `tbPairwiseInteraction.mat`: MATLAB file containing contact records.

3. **Contact Duration and Frequency Statistics**:
   Run `StaticIntervalsCreator.m`:
   ```matlab
   StaticIntervalsCreator;
   ```
   Output:
   * `staticInterval.txt`: Detailed contact duration and frequency information.
   * `AggregatedDuration.mat`: Aggregated contact duration statistics.
   * `AggregatedNumber.mat`: Aggregated contact count statistics.

4. **Filter Contact Information**:
   Run ``PartlyAggregated.m``:
   ```matlab
   PartlyAggregated;
   ```
   Output:
   * Filtered contact records based on user-specified criteria.

### 2.4 FudanWiFi13
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamic/blob/master/TeNet/FudanWiFi13/FudanWiFi13_framework.png" width="950px">

Using the provided scripts, we processed the **FudanWiFi13** dataset, containing:
* **12,433 users**
* **1,124,026 login events**
* **206,892 user pairs**
* **2,331,597 interactions**

The generated files are as follows:
1. ``dyadicdata_TimeINDetail.txt``:
   * All records of user pairs.
   * Format: ``User1, User2, StartTime (year, month, day, hour, minute), weekday, duration (min), AP information``.
2. ``singleperson_For_regularSamples_anonymous_TimeINDetail.txt``:
   * All records of individual users.
   * Format: ``User ID, StartTime (year, month, day, hour, minute), weekday, duration (min), AP information``.
3. ``dyadicID.txt``:
   * Mapping of user pair IDs to corresponding users.
   * Format: ``User Pair ID, User1, User2``.
4. ``intervalBeyondDay.txt``:
   * Event time intervals across different days.
   * Format: ``User Pair ID, time_interval (min)``.
5. ``intervalWithinDay.txt``:
   * Event time intervals within the same day.
   * Format: ``User Pair ID, time_interval (min)``.
6. ``dailyNetwork.txt``:
   * Daily network data.
   * Format: ``User1, User2, Day number``.
7. ``aggregated_Network.txt``:
   * Aggregated network node statistics.
   * Format: ``Node (User ID), occurrence count, degree``.

These files provide a comprehensive dataset for researchers to analyze and study.

## 3. SubTrack
### 3.1 Background Information
In modern urban transit systems, understanding passenger travel patterns is critical for improving subway operations and enhancing service quality. The **SubTrack** project simulates passenger trajectories within a subway system based on detailed smart card data. By analyzing passenger card swipe times and matching them to train schedules, the framework reconstructs the specific paths passengers take through the subway network.

This tool provides valuable insights into passenger mobility patterns, travel behavior, and train utilization. By leveraging these insights, transit authorities can optimize subway schedules, enhance passenger experience, and conduct analyses for public health or operational planning. The track_assignment framework offers a robust method for processing and aligning these datasets to support these goals effectively.

### 3.2 Key Features
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamic/blob/master/SubTrack/SubTrack_Framework.png" width="950px">

**SubTrack** includes the following core functionalities:

#### 1) Data Import and Preprocessing:
* **Passenger Data Processing**:
  * Import passenger data from CSV files (e.g., ``stoptime_passengers/{date_id}``).
  * Format data files with columns: ``Card Number``, ``Passage Time``.
  * Perform unit conversion if necessary (e.g., ``timestamps``).
* **Subway Data Processing**:
  * Import subway operation data from CSV files (e.g., ``subways_merged/{week_id}``).
  * Format data files with columns: ``Train Number``, ``Passage Time``.

#### 2) Passenger-Train Matching:
* Match passengers to specific trains based on their passage time and the schedule of subway operations.
* Passenger belongs to ``Train(i)`` if:
  
  ``Train(i-1)_Time < Passenger_Time <= Train(i)_Time``
  
  * ``Train(i-1)/(i)_Time``: Subway passage time from microscope subway data.
  * ``Passenger_Time``: Passenger passage time from microscope passenger data.

### 3.3 How to Run
#### 1) Prerequisites:
* Python 3.7 or later.
* Required libraries: ``pandas``, ``numpy`` (``install with pip install pandas numpy``).
* Input data files in the following formats:
  * **Passenger Data**: CSV files named ``{line}_{direction}_stoptime.csv`` containing columns:
    * Card Number.
    * Passage Time.
  * **Subway Data**: CSV files named ``{line}_{direction}_merged.csv`` containing columns:
    * Train Number
    * Passage Time.

#### 2) Setup:
1. Clone or download this repository:
```bash
git clone https://github.com/CAN-Lab-Fudan/ViralDynamic/tree/master/SubTrack.git
```

2. Place the input data files in the appropriate directories:
   * Passenger data: ``stoptime_passengers/{date_id}``.
   * Subway data: ``subways_merged/{week_id}``.

3. Ensure that the output directory ``matched/{date_id}`` exists or will be created during script execution.

#### 3) Running the Scripts:
Follow these steps to run the tool:

1. **Data Import and Preprocessing**:
   Run the script with the following command:
   ```Python
   python track_assignment.py <date_id> <week_id>
   ```
   * ``<date_id>``: The specific date for folder containing passenger data (e.g., ``0417``, ``0423``).
   * ``<week_id>``: The type of train schedule corresponding to the folder (e.g., ``workday``, ``weekend``).

2. **Output**:
   * Matched Passenger-Train Data (CSV files):
     * Located in ``matched/{date_id}``.
     * Filename format: ``{line}_{direction}_matched.csv``.
     * Columns: ``Card Number``, ``Train Number``, ``Passage Time``.
   
## 4. TaHiP
### 4.1 Background Information
Predicting future dynamics on networks, especially hypergraphs with unknown topology, is a challenging task due to the complexity of higher-order interactions. This project introduces **TaHiP** (Topology-Agnostic-Higher-Order Dynamics Prediction), a novel algorithm designed to predict dynamics on hypergraphs without requiring complete topology information. By leveraging observed nodal states to train a surrogate matrix, **TaHiP** accurately predicts future states on hypergraphs. Experiments show that **TaHiP** outperforms Transformer-based models in both synthetic and real-world settings, offering a robust solution for dynamic prediction tasks.

Our paper is available at [this link](https://ieeexplore.ieee.org/document/10794522). Citations are welcome.

```bibtex
@ARTICLE{10794522,
  author={Zhou, Zili and Li, Cong and Mieghem, Piet Van and Li, Xiang},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers}, 
  title={Predicting Higher-Order Dynamics With Unknown Hypergraph Topology}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Power system dynamics;Topology;Heuristic algorithms;Mathematical models;Predictive models;Accuracy;Prediction algorithms;Vectors;Integrated circuit modeling;Training;Nonlinear system;dynamics on networks;predicting higher-order dynamics;contagion;hypergraph},
  doi={10.1109/TCSI.2024.3513406}}
```
### 4.2 Key Features
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamic/blob/master/TaHiP/TaHiP_framework.png" width="650px">

**TaHiP** includes the following core functionalities:

#### 1) Flexible Hypergraph Formation:
* **Synthetic Hypergraph Generation**: Create hypergraphs with adjustable edge size distributions (e.g., uniform, Poisson, or custom-defined).
* **Real-World Hypergraph Support**: Load and process hypergraph datasets from social networks, biological systems, or legislative processes.

#### 2) Higher-Order Dynamics Modeling
* Implements two key models for predicting dynamics:
  * **Product Model**: Capture nonlinear interactions between nodes within hyperedges.
  * **Threshold Model**: Simulate dynamics where node state changes are triggered by threshold-based hyperedge activations.
* Suitable for modeling real-world processes such as social contagion, disease spread, and cooperative behaviors.

#### 3) Topology-Agnostic Prediction:
* Predict higher-order dynamics without prior knowledge of the hypergraph topology.
* Train a **surrogate matrix** from observed nodal states, enabling predictions in systems with unknown or complex structures.

#### 4) Analysis of Hyperedge Size Effects
* Analyze how **mean hyperedge size** and its **distribution** affect prediction performance:
  * Larger hyperedges increase prediction difficulty.
  * Incorporating known hyperedge size distributions improves accuracy and reduces errors.

### 4.3 How to Run
#### 1) Prerequisites:
* Python: ``3.8 <= Python <= 3.13``
* Required libraries: ``matplotlib``, ``numpy``, ``scikit-learn``, ``torch==1.13.0``  (``install with pip install matplotlib numpy scikit-learn torch==1.13.0``).

#### 2) Setup
Clone or download this repository:
```bash
git clone https://github.com/CAN-Lab-Fudan/ViralDynamic/tree/master/TaHiP.git
```
#### 3) Running the Scripts:

Follow these steps to run the tool:

1. **Data Import and Preprocessing**:

    This step generates hypergraph structures from synthetic or real-world datasets.

    Run the following command:

   ```Python
   python run_hypergraph_formation.py
   ```

   * **Input**: Raw data (e.g., hypergraph node and edge lists or configuration parameters for synthetic generation).
   * **Output**: Hypergraph structure saved in a specified format.

2. **Training and Forecasting**

    This step trains the surrogate matrix on observed nodal states and forecasts future dynamics.

    Run the following command:
   ```Python
   python run_product_model_copredict.py
   ```

   * **Input**: Hypergraph structure, initial nodal states.
   * **Output**: Predicted nodal states over time.

### 4.4 Acknowledgement
We appreciate Professor Austin R. Benson and his team a lot for their valuable hypergraph datasets provided in https://www.cs.cornell.edu/~arb/data/

## 5. ε-SIS
### 5.1 Background Information
The **ε-SIS** model is an extension of the classical Susceptible-Infected-Susceptible (SIS) framework, designed to better capture the complexity of real-world infection dynamics. By introducing a spontaneous infection mechanism governed by an external infection rate ε, the model accounts for infections arising from environmental or external factors. This enhancement allows the **ε-SIS** model to simulate persistent and reoccurring infections, even in the absence of direct neighbor-to-neighbor transmission. Employing an event-driven, continuous-time method, this project efficiently simulates disease spread in large-scale networks, offering fine-grained control over infection dynamics and generating insightful visualizations of epidemic progression.

Our paper is available at [this link](https://link.aps.org/doi/10.1103/PhysRevE.86.026116). Citations are welcome.

```bibtex
@article{PhysRevE.86.026116,
  title = {Susceptible-infected-susceptible model: A comparison of $N$-intertwined and heterogeneous mean-field approximations},
  author = {Li, Cong and van de Bovenkamp, Ruud and Van Mieghem, Piet},
  journal = {Phys. Rev. E},
  volume = {86},
  issue = {2},
  pages = {026116},
  numpages = {9},
  year = {2012},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.86.026116},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.86.026116}
}
```

### 5.2 Key Features

**ε-SIS** includes the following core functionalities:

#### 1) Disease Spreading Modeling:
* **Neighbor-to-Neighbor** Transmission:
  * Simulate disease spread through direct interactions between neighboring nodes.
  * Model infection events based on an infection rate β with transmission times sampled from an exponential distribution.
* **Spontaneous Infection**:
  * Incorporate infections caused by **external environmental** or **stochastic factors**.
  * Model infection events driven by a spontaneous infection rate ε ensuring realistic persistence of disease.

#### 2) Event-Driven Continuous-Time Simulation:
* **Efficient Event Processing**:
  * Use a **priority queue** to dynamically manage and process infection, recovery, and spontaneous infection events.
  * Ensure simulation scalability for large-scale networks by updating only the nodes involved in an event.
* **Stochastic Timing**:
  * Simulate events based on **exponential distributions**, reflecting the probabilistic nature of real-world dynamics.

#### 3) Infection Dynamics Tracking:
* Record the number of infected nodes over time, providing detailed insights into the progression of the disease.
* Generate time-series data for post-simulation analysis and visualization.

### 5.3 How to Run
#### 1) Prerequisites:
* Python 3.7 or later.
* Required libraries: ``networkx``, ``numpy``, ``matplotlib``, ``tqdm``, ``heapq`` (``install with pip install networkx numpy matplotlib tqdm heapq``).

#### 2) Setup
Clone or download this repository:
```bash
git clone https://github.com/CAN-Lab-Fudan/ViralDynamic/tree/master/ε-SIS.git
```

#### 3) Running the Scripts:
Execute the main simulation script:
```Python
python ε-SIS.py
```

#### 4) Example Usage
Below is an example workflow to simulate a ER network of 500 nodes with specific infection and recovery rates。 The specific adjustable parameters in ``ε-SIS.py``:
```Python
N = 100                          # Number of nodes
p = 2 * np.log(N) / N            # Connection probability 
G = nx.erdos_renyi_graph(N, p)   # Graph construction
delta = 1                        # Recovery rate
epsilon = 0.001                  # Spontaneous infection rate
beta = 0.6 * delta               # Infection rate
max_time = 30                    # Maximum Iteration Steps
```

## 6. EvoVax
### 6.1 Background Information
Game theory provides a powerful framework for modeling strategic decision-making in situations where individuals' actions are interdependent, such as in epidemic control and vaccination strategies. Traditional immune strategies often assume uniform individual attitudes toward epidemics, epidemic-related information, and vaccination, with less attention to individual heterogeneity and psychological factors. **EvoVax** addresses this by incorporating prospect theory and a two-layer network model to capture how individual perceptions and social factors influence vaccination decisions. **EvoVax** offers more personalized, accurate, and flexible immune strategies, improving epidemic control in dynamic environment.

Our paper is available at [this link](https://ieeexplore.ieee.org/document/10124338). Citations are welcome.

```bibtex
@ARTICLE{10124338,
  author={Li, Cong and Dai, Jin-Ying and Li, Xiang},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={Imperfect Vaccination Evolutionary Game Incorporating Individual Social Difference and Subjective Perception}, 
  year={2024},
  volume={11},
  number={2},
  pages={2369-2382},
  keywords={Epidemics;Vaccines;Behavioral sciences;Games;Costs;Psychology;Game theory;Epidemic spreading;prospect theory (PT);psychological cognition;social difference;vaccination game},
  doi={10.1109/TCSS.2023.3271894}}
```

### 6.2 Key Features
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamic/blob/master/EvoVax/EvoVax_framework.png" width="950px">

**EvoVax** includes the following core functionalities:

#### 1) Coupled-network Modeling:
* **Physical Contact Layer**: Construct a synthetic network to characterise physical connections between individuals.
* **Information Layer**: Construct a synthetic network to simulate messaging.

#### 2) Information Diffusion and Epidemic Spreading:
* Simulate the diffusion of information and its effect on epidemic transmission.
* Model how social factors, such as information exposure and individual trust, influence behavior during an epidemic, including protective actions and vaccination uptake.
* Incorporate dynamic individual protection levels based on personal perceptions, behavior changes, and exposure to information.

#### 3) Vaccination Strategy Updates:
* Calculate the payoff of individuals based on pairwise comparison rules for strategy updates.
* Model iteration to steady state yields immunological proportions.

### 6.3 How to Run
#### 1) Prerequisites:
* Python 3.8 or later.
* Required libraries: ``networkx``, ``numpy``, ``matplotlib``, ``math``, ``random``  (``install with pip install networkx, numpy, matplotlib, math, random``).

#### 2) Setup
Clone or download this repository:
```bash
git clone https://github.com/CAN-Lab-Fudan/ViralDynamic/tree/master/EvoVax.git
```

#### 3) Running the Scripts:

Follow these steps to run the tool:

1. **Generation Data**:

   This step generates network data and save it as a txt file or reference the file to generate related network data.

   Run the following command:

   ```Python
   python Model_Code_Opt_Data.py
   ```

  * **Output**: `network_data.txt` includes:
    *  Nodes
    *  Network link.

2. **Individual Protection Level Calculation**:

   This step generates the individual protection level.

   Run the following command:

   ```Python
   python ProtLevLambDel_code.py
   ```

  * **Output**:
    * `protection_levels.txt`: Individual protection level.

3. **Information Diffusion and Epidemic Spreading**:

   This step simulates the dynamic of information diffusion and epidemic spreading

   Run the following command:

   ```Python
   python McRhIRhA_code_end.py
   ```

  * **Input**: `network_data.txt` and `protection_levels.txt`
  * **Output**:
    * ``PA.txt``: The fraction of aware individuals.
    * ``PI.txt``: The fraction of infected individuals.

4. **Vaccination Strategy Updates**:

   This step simulates an immune policy update.

   Run the following command:

   ```Python
   python Transi_inf_epi_upda.py
   ```

  * **Input**: `network_data.txt`, `protection_levels.txt`, `PA.txt`` and ``PI.txt``.
  * **Output**:
    * ``result_x.txt``: The fraction of vaccinated individuals.
    * ``result_I.txt``: The fraction of infected individuals.

## 7. Future Work
As **ViralDynamic** continues to evolve, our primary goal is to enhance its capabilities and expand its applicability in epidemic modeling, data analysis, and decision-making processes. Below are some key areas we plan to focus on in the future:
* **Integration of Multi-Source Data**: We aim to expand **ViralDynamic**' ability to integrate diverse data sources, including real-time data from wearable devices, health monitoring systems, and social media platforms. This will improve the accuracy of epidemic spread predictions by incorporating more granular, real-time user behavior data.

* **Personalized Epidemic Modeling**: **ViralDynamic** will focus on personalizing epidemic simulations to account for heterogeneous population characteristics. This will include simulating individual-level responses to public health measures, such as vaccination campaigns or isolation, based on factors like health status, risk perception, and behavioral tendencies.

* **Advanced Machine Learning Algorithms for Prediction**: We plan to incorporate cutting-edge machine learning techniques, including reinforcement learning and deep learning models, to enhance the framework's prediction accuracy. By integrating these models, **ViralDynamic** will be able to forecast epidemic dynamics with even greater precision, enabling more proactive public health interventions.

* **Collaborative and Scalable Solutions**: We envision **ViralDynamic** as a collaborative platform that allows for easy integration with other public health and epidemiology tools. By providing scalable solutions for large-scale epidemic simulations, **ViralDynamic** will contribute to global epidemic preparedness and response efforts.
