# Datasets description

Get the datasets from the links provided in the root README.md.

All the following descriptions come from Kaggle links that can be found in the root README.md.

# Boston Housing

Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are deﬁned as follows (taken from the UCI Machine Learning Repository1): 

- CRIM: per capita crime rate by town.
- ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX: nitric oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to ﬁve Boston employment centers
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population
- MEDV: Median value of owner-occupied homes in $1000s

# Energy Efficiency

We perform energy analysis using 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. We simulate various settings as functions of the afore-mentioned characteristics to obtain 768 building shapes. The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses. It can also be used as a multi-class classification problem if the response is rounded to the nearest integer.

Attribute Information:

The dataset contains eight attributes (or features, denoted by X1…X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.

Specifically:

- X1 Relative Compactness
- X2 Surface Area
- X3 Wall Area
- X4 Roof Area
- X5 Overall Height
- X6 Orientation
- X7 Glazing Area
- X8 Glazing Area Distribution
- y1 Heating Load
- y2 Cooling Load

## Relevant Papers:

A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012

## Citation Request:

A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012 (the paper can be accessed from [Web Link])

For further details on the data analysis methodology:
A. Tsanas, 'Accurate telemonitoring of Parkinsonâ€™s disease symptom severity using nonlinear speech signal processing and statistical machine learning', D.Phil. thesis, University of Oxford, 2012 (which can be accessed from [Web Link])

# Naval Propulsion Plant

## Maintenance of Naval Propulsion Plants Data Set
Predicting Gas Turbine propulsion plant's decay state coefficient

## Motivation:

In this case-study, we generate a predictive model to predict of decay state of rotating equipment using

## Abstract:

Dataset source:  
- [UCI](http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants)
-  [Kaggle](https://www.kaggle.com/elikplim/maintenance-of-naval-propulsion-plants-data-set)

Data have been generated from a sophisticated simulator of a Gas Turbines (GT), mounted on a Frigate characterized by a COmbined Diesel eLectric And Gas (CODLAG) propulsion plant type.

## Problem Statement:

The experiments have been carried out by means of a numerical simulator of a naval vessel (Frigate) characterized by a Gas Turbine (GT) propulsion plant. The different blocks forming the complete simulator (Propeller, Hull, GT, Gear Box and Controller) have been developed and fine tuned over the year on several similar real propulsion plants. In view of these observations the available data are in agreement with a possible real vessel.

In this release of the simulator it is also possible to take into account the performance decay over time of the GT components such as GT compressor and turbines.

The propulsion system behaviour has been described with this parameters:

    Ship speed (linear function of the lever position lp).
    Compressor degradation coefficient kMc.
    Turbine degradation coefficient kMt. so that each possible degradation state can be described by a combination of this triple (lp,kMt,kMc).

The range of decay of compressor and turbine has been sampled with an uniform grid of precision 0.001 so to have a good granularity of representation. In particular for the compressor decay state discretization the kMc coefficient has been investigated in the domain [1; 0.95], and the turbine coefficient in the domain [1; 0.975]. Ship speed has been investigated sampling the range of feasible speed from 3 knots to 27 knots with a granularity of representation equal to tree knots. A series of measures (16 features) which indirectly represents of the state of the system subject to performance decay has been acquired and stored in the dataset over the parameter's space.

## References:

 - [Simonwenkel](https://www.simonwenkel.com/2019/04/19/revisitingML-naval-propulsion.html)
 - [Researchgate](https://www.researchgate.net/publication/245386997_Real-time_simulation_of_a_COGAG_naval_ship_propulsion_system)
 - [Linkedin Article](https://www.linkedin.com/pulse/gas-turbine-compressor-decay-state-coefficient-john-kingsley/?trackingId=5S5swf3uTqCizwyGWxxSIw%3D%3D)

# Protein Structures

## Context

This is a protein data set retrieved from Research Collaboratory for Structural Bioinformatics (RCSB) Protein Data Bank (PDB).

The PDB archive is a repository of atomic coordinates and other information describing proteins and other important biological macromolecules. Structural biologists use methods such as X-ray crystallography, NMR spectroscopy, and cryo-electron microscopy to determine the location of each atom relative to each other in the molecule. They then deposit this information, which is then annotated and publicly released into the archive by the wwPDB.

The constantly-growing PDB is a reflection of the research that is happening in laboratories across the world. This can make it both exciting and challenging to use the database in research and education. Structures are available for many of the proteins and nucleic acids involved in the central processes of life, so you can go to the PDB archive to find structures for ribosomes, oncogenes, drug targets, and even whole viruses. However, it can be a challenge to find the information that you need, since the PDB archives so many different structures. You will often find multiple structures for a given molecule, or partial structures, or structures that have been modified or inactivated from their native form.

## Content

There are two data files. Both are arranged on "structureId" of the protein:

    pdbdatano_dups.csv contains protein meta data which includes details on protein classification, extraction methods, etc.

    data_seq.csv contains >400,000 protein structure sequences.

# Wine Quality 
The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones).

This dataset is also available from the UCI machine learning repository, https://archive.ics.uci.edu/ml/datasets/wine+quality , I just shared it to kaggle for convenience. (If I am mistaken and the public license type disallowed me from doing so, I will take this down if requested.)

## Content

For more information, read [Cortez et al., 2009].

Input variables (based on physicochemical tests):

1 - fixed acidity

2 - volatile acidity

3 - citric acid

4 - residual sugar

5 - chlorides

6 - free sulfur dioxide

7 - total sulfur dioxide

8 - density

9 - pH

10 - sulphates

11 - alcohol

Output variable (based on sensory data):

12 - quality (score between 0 and 10)
## Tips

What might be an interesting thing to do, is aside from using regression modelling, is to set an arbitrary cutoff for your dependent variable (wine quality) at e.g. 7 or higher getting classified as 'good/1' and the remainder as 'not good/0'.
This allows you to practice with hyper parameter tuning on e.g. decision tree algorithms looking at the ROC curve and the AUC value.
Without doing any kind of feature engineering or overfitting you should be able to get an AUC of .88 (without even using random forest algorithm)

KNIME is a great tool (GUI) that can be used for this.

1 - File Reader (for csv) to linear correlation node and to interactive histogram for basic EDA.

2- File Reader to 'Rule Engine Node' to turn the 10 point scale to dichtome variable (good wine and rest), the code to put in the rule engine is something like this:

    $quality$ > 6.5 => "good"
    TRUE => "bad"

    3- Rule Engine Node output to input of Column Filter node to filter out your original 10point feature (this prevent leaking)

    4- Column Filter Node output to input of Partitioning Node (your standard train/tes split, e.g. 75%/25%, choose 'random' or 'stratified')

    5- Partitioning Node train data split output to input of Train data split to input Decision Tree Learner node and

    6- Partitioning Node test data split output to input Decision Tree predictor Node

    7- Decision Tree learner Node output to input Decision Tree Node input

    8- Decision Tree output to input ROC Node.. (here you can evaluate your model base on AUC value)

## Inspiration

Use machine learning to determine which physiochemical properties make a wine 'good'!
## Acknowledgements

This dataset is also available from the UCI machine learning repository, https://archive.ics.uci.edu/ml/datasets/wine+quality , I just shared it to kaggle for convenience. (I am mistaken and the public license type disallowed me from doing so, I will take this down at first request. I am not the owner of this dataset.

Please include this citation if you plan to use this database: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
## Relevant publication

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009. 

# Yacht Hydrodynamics

Delft data set, used to predict the hydrodynamic performance of sailing yachts from dimensions and velocity.

## Source

Creator:

Ship Hydromechanics Laboratory, Maritime and Transport Technology Department, Technical University of Delft.

Donor:

Dr Roberto Lopez
E-mail: roberto-lopez '@' users.sourceforge.net
## Data Set Information:

Prediction of residuary resistance of sailing yachts at the initial design stage is of a great value for evaluating the shipâ€™s performance and for estimating the required propulsive power. Essential inputs include the basic hull dimensions and the boat velocity.

The Delft data set comprises 308 full-scale experiments, which were performed at the Delft Ship Hydromechanics Laboratory for that purpose.
These experiments include 22 different hull forms, derived from a parent form closely related to the â€˜Standfast 43â€™ designed by Frans Maas.

## Attribute Information:

Variations concern hull geometry coefficients and the Froude number:

    Longitudinal position of the center of buoyancy [LC], adimensional.
    Prismatic coefficient [PC], adimensional.
    Length-displacement ratio [L/D], adimensional.
    Beam-draught ratio [B/Dr], adimensional.
    Length-beam ratio [L/B], adimensional.
    Froude number [Fr], adimensional.

The measured variable is the residuary resistance per unit weight of displacement:

    Residuary resistance per unit weight of displacement [Rr], adimensional.

## Relevant Papers:

J. Gerritsma, R. Onnink, and A. Versluis. Geometry, resistance and stability of the delft systematic yacht hull series. In International Shipbuilding
Progress, volume 28, pages 276â€“297, 1981.

I. Ortigosa, R. Lopez and J. Garcia. A neural networks approach to residuary resistance of sailing
yachts prediction. In Proceedings of the International Conference on Marine Engineering MARINE
2007, 2007.

## Citation Request:

Lopez, R. (2013). Yacht Hydrodynamics Data Set, UCI Machine Learning Repository, avaliable in: https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics. Irvine, CA: University of California, School of Information and Computer Science.