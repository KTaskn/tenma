data {
    int N;
    real X[N];
    int <lower=0, upper=1> Y[N];
}

parameters {
    real a;
    real b;
}

model {
    a ~ normal(1.0, 0.05);
    for (n in 1:N){
        Y[n] ~ bernoulli(inv_logit(a * X[n] + b));
    }
}