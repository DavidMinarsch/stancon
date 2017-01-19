/*
This model says that there is some probability `theta` that `y`
is zero and probability `1 - theta` that `y` is positive. 
Conditional on observing a positive `y`, we use a truncated 
Poisson

```
y[n] ~ Poisson(lambda) T[1, U];
```
where `T[1,U]` indicates truncation with lower bound `1` and upper bound `U`, 
which for simplicity we'll _assume_ is `max(y)`.
*/
/*
functions {
  real hurdle(real a) {
    return a;
  }
}
*/
data {
  int<lower=1> N;
  int y[N];
}
transformed data {
  int U = max(y);
}
parameters {
  //real<lower=1, upper=U> lambda;
  real<lower=1> lambda;
  real<lower=0, upper=1> theta;
  //real<lower=0, upper=1> nonzero;
}
model {
  //lambda ~ normal(0, 5);
  //nonzero ~ bernoulli(theta);
  //y ~ poisson(lambda * bern_param);
  //y ~ nonzero ? poisson(lambda) : 0;
  
  // Prior for lambda, as an example
  lambda ~ exponential(0.2);
  
  for (n in 1:N) {
    if (y[n] == 0) {
      // Add up terms on log scale that define joint prob
      // target = reserved word, distribution we're targeting
      target += log(theta); // probability that y == 0 times whatever?
      // increment_log_prob deprecated
    }
    else {
      target += log1m(theta);
      // log(1 - theta) so (1 - theta) is multiplied
      // by log poisson
      y[n] ~ poisson(lambda) T[1, U];
    }
  }
}
generated quantities {
  /*
  int y_rep[N];
  for (n in 1:N)
    y_rep[n] = poisson_rng(lambda);
  */
  int y_rep[N];
  for (n in 1:N) {
    if (bernoulli_rng(theta)) {
      y_rep[n] = 0;
    }
    else {
      int w;
      w = poisson_rng(lambda);
      while (w == 0 || wc > U)
        w = poisson_rng(lambda);
      
      y_rep[n] = w;
    }
  }
}