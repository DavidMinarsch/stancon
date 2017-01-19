data {
  // Dimensions; number of observations
  // lower=1 is optional; helpful to sanity check that there is at least 1 observation
  int<lower=1> N;
  int<lower=1> K; // Number of features?
  
  // Variables
  // Example: predict weight from height and age (K = 2)
  // So X is N values for height, N values for age = N x 2 = N x K
  matrix[N,K] X;
  
  // N observations
  vector[N] y;
}
// constraints required in parameters block when they're logically necessary
// e.g. true by definition, not stuff we think
parameters {
  // Can be convenient to declare coeff and intercept separately
  // if e.g. want different prior on the intercept
  real alpha; // intercept/constant term
  vector[K] beta; // coeff for height, coeff for age
  // Can't have sigma < 0 for this to be a valid model;
  // stan needs to know this
  // under the hood, stan takes the log of sigma, when you declare lower=0
  // if not, stan won't take log of sigma, we could get negative values for sigma
  // and hell hell breaks loose?
  real<lower=0> sigma; // noise/error
}
model {
  // y is vector of length N
  // instead of loop, we mean: y1 ~ normal(X vars for first observation)
  // X * beta: matrix times vector; don't need special matrix mult symbols
  // ...because stan knows what X and beta are
  // N x k * k x 1 = N x 1
  // alpha gets added to each of the N elements of the vector
  y ~ normal(X * beta + alpha, sigma);
  // Alternatively, slower: for (n in 1:N) y[n] ~ normal(X[n,] * beta)
  // What's really going on is addition, not multiplication
  // don't read this as y is drawn from normal distribution
  // actually: adding up N logarithms of the normal distribution
  // evaluating it, rather than drawing a number from it
  // we already know values of y, why would we draw?
  // we would want to draw beta/alpha, but not from this
  
  // priors
  // what else do we know alpha,beta
  // flat, uniform if omitted
  // by default: uniform distribution from -infinity to infinity (not actually a probability distribution)
  // some models, you can get a valid posterior even with an improper prior; but sometimes you won't get valid posterior
  // why is the default automatically uniform? adding things on the log scale,
  // nothing added to log prob -- adding 0
  alpha ~ normal(0, 10); // bell curve, std dev 10
  // under the hood: not drawing alpha from this distribution
  // rather: this will get evaluating hundreds/thousands of times
  // each time: get single value for alpha, K values for beta, etc.
  // so set of 1000 values
  // by the time we get into the model block, values for alpha/beta
  // those values might get rejected (ref MCMC, proposed value)
  // proposed values are present; don't need to draw value for alpha,
  // **rather have value for alpha, and evaluate it on normal(0, 10) or something?
  // prior distribution of alpha is indeed summarized by normal(0, 10),
  // just not being drawn
  
  // note that this is vectorized
  // assuming these betas are indepedent of each other
  // if you had correlations/covariance, you'd use multi_normal
  beta ~ normal(0, 10);
  
  // positive half of cauchy because of lower=0
  sigma ~ cauchy(0, 10);
}
// run at every MCMC iteration; at each iteration alpha/beta/sigma have
// whatever the proposed value was
// e.g. what does our model think our data/simulate data
// involved in cross-validation
// NOTE: this is not sampling from the posterior; samples from posterior = alpha/beta/sigma
// VERSUS: this is the posterior predictive distribution: USE posterior estimates of alpha/beta/sigma to make predictions for data
// rep: replicate data that i already have, not predict for new data points
// If had X1 and X2 data, could use X2 here to make predictions using a held out
// set of data
// COULD do this in R using N x 1000 matrix instad of N x 1 here at each iteration
// QUESTION: why normal_rng
// ANSWER: what about sigma? y not perfectly explained by alpha/beta.
// mathematically correct way to propogate all the uncertainty you have into your observations
// uncertainty in (a) parameters and (b) data generating process
generated quantities {
  // Draw y from the normal distribution with alpha/beta/sigma plugged in
  vector[N] y_rep;
  // RNGs not vectorized (yet?)
  for (n in 1:N)
    y_rep[n] = normal_rng(X[n, ] * beta + alpha, sigma); // X[n,] = entire nth row of X
    // y_rep[n] = normal_rng(X[n,] * beta + alpha, sigma);
}