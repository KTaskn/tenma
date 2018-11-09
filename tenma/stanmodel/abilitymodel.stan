data {
    // data length
    int N;
    int <lower=1, upper=18> X_R[N, 4];
    int <lower=1, upper=10> X_G[N, 4];
    int <lower=1, upper=10> X_RACE[N];
    real <lower=0> X_KISYU[N];
    int <lower=0, upper=1> Y[N];
}

parameters {
    real p[4];
    real bias;
    real g_a;
    real g_b;
    real k_a;
}

model {
    p ~ normal(1.0, 0.1);
    k_a ~ normal(1.0, 0.1);
    bias ~ normal(-5.0, 0.01);

    for (n in 1:N){
        Y[n] ~ bernoulli(inv_logit(
            p[1] * (1.0 / X_R[n, 1]) * inv_logit(g_a * (X_RACE[n] - X_G[n, 1]) + g_b)
            + p[2] * (1.0 / X_R[n, 2]) * inv_logit(g_a * (X_RACE[n] - X_G[n, 2]) + g_b)
            + p[3] * (1.0 / X_R[n, 3]) * inv_logit(g_a * (X_RACE[n] - X_G[n, 3]) + g_b)
            + p[4] * (1.0 / X_R[n, 4]) * inv_logit(g_a * (X_RACE[n] - X_G[n, 4]) + g_b)
            + k_a * X_KISYU[n]
            + bias
        ));
    }
}