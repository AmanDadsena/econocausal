# References

Reading list for the project, with a note on why each one matters here.

## Method

- **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., Robins, J. (2018).**
  *Double/debiased machine learning for treatment and structural parameters.*
  The Econometrics Journal, 21(1), C1–C68.
  The original DML paper. Explains the residual-on-residual idea, Neyman orthogonality
  and cross-fitting, which are the base of the elasticity estimator in this project.

- **Wager, S., Athey, S. (2018).**
  *Estimation and inference of heterogeneous treatment effects using random forests.*
  Journal of the American Statistical Association, 113(523), 1228–1242.
  Causal forests. Used for elasticities that differ by segment (category, season, country).

- **Athey, S., Tibshirani, J., Wager, S. (2019).**
  *Generalized random forests.* The Annals of Statistics, 47(2), 1148–1178.
  The framework behind `CausalForestDML` in EconML.

- **Frisch–Waugh–Lovell theorem.**
  Any econometrics textbook. Shows that partialling out the controls from both the
  outcome and the treatment gives the same coefficient as the full regression. DML is the
  ML version of this idea.

## Libraries

- **EconML** (PyWhy): https://github.com/py-why/EconML
  `LinearDML` and `CausalForestDML` are used for the main estimates.
- **DoubleML**: https://docs.doubleml.org
  An alternative DML implementation, useful for cross-checking results.
- **DoWhy** (PyWhy): https://github.com/py-why/dowhy
  Ideas for the refutation tests (placebo treatment, random common cause, data subsets).

## Learning resources

- **Chernozhukov, V., Hansen, C., Kallus, N., Spindler, M., Syrgkanis, V.**
  *Applied Causal Inference Powered by ML and AI.* https://causalml-book.org
  Practical chapters on DML with code.
- **Facure, M.** *Causal Inference for the Brave and True.*
  https://matheusfacure.github.io/python-causality-handbook/
  Clear introduction to confounding and DML, including a price elasticity example.

## Data

- **Chen, D. (2019).** *Online Retail II.* UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/502/online+retail+ii
  Transactions of a UK online retailer, Dec 2009 – Dec 2011. Main dataset.

## Pricing

- With constant elasticity θ < −1 and unit cost c, the profit-maximising price is
  p* = c · θ / (1 + θ) (the inverse elasticity / Lerner rule). The optimizer uses this
  as a reference but searches within business limits instead of applying it directly.
