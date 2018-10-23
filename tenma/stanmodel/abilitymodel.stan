data {
    // data length
    int N;
    // rank kind
    int R;
    int G;
    int X_R[N, 4];
    int X_G[N, 4];
    int X_RACE[N];
    int Y[N];
}

parameters {
    real p[R, G];
    real race_p[G];

    real p_base_r[R];
}

model {
    for (g in 1:G){
        race_p[g] ~ normal(0.0, 3.0);
    }

    for (r in 1:R){
        p_base_r[r] ~ normal(1 / r, 0.05);
    }

    for (r in 1:R){
        for (g in 1:G){
            p[r, g] ~ normal(p_base_r[r], 0.1);
        }
    }
    for (n in 1:N){
        Y[n] ~ bernoulli(inv_logit(
            p[X_R[n, 1], X_G[n, 1]]
            + p[X_R[n, 2], X_G[n, 2]]
            + p[X_R[n, 3], X_G[n, 3]]
            + p[X_R[n, 4], X_G[n, 4]]
            + race_p[X_RACE[n]]
        ));
    }
}