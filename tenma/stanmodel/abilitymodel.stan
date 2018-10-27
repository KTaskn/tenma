data {
    // data length
    int N;
    int <lower=1, upper=18> X_R[N, 4];
    int <lower=1, upper=10> X_G[N, 4];
    int <lower=1, upper=10> X_RACE[N];
    int <lower=0, upper=1> Y[N];
}

parameters {
    real p[4];
    real bias;
    real r_a;
    real r_bias;
}

model {
    p ~ normal(1.0, 0.1);
    bias ~ normal(-4.0, 0.01);
    for (n in 1:N){
        Y[n] ~ bernoulli(inv_logit(
            p[1] * (1.0 / X_R[n, 1]) * inv_logit(r_a * (X_RACE[n] - X_G[n, 1]) + r_bias)
            + p[2] * (1.0 / X_R[n, 2]) * inv_logit(r_a * (X_RACE[n] - X_G[n, 2]) + r_bias)
            + p[3] * (1.0 / X_R[n, 3]) * inv_logit(r_a * (X_RACE[n] - X_G[n, 3]) + r_bias)
            + p[4] * (1.0 / X_R[n, 4]) * inv_logit(r_a * (X_RACE[n] - X_G[n, 4]) + r_bias)
            + bias
        ));
    }
}