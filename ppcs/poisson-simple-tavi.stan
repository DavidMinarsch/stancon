data {
  int<lower=1> N;
  int y[N];
}
parameters {
  real<lower=0> lambda;
}
model {
  lambda ~ normal(0, 5);
  y ~ poisson(lambda);
}
generated quantities {
  int y_rep[N];
  for (n in 1:N)
    y_rep[n] = poisson_rng(lambda);
}