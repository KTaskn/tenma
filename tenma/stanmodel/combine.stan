data {
    // data length
    int N;
    int D;
    real X[N, D];
    real Y[N];
    real O[N];
    real P;
}

parameters {
    real W_1[D];
    real W_2[D];
    real B[D];
    real bias;
}

model {
    real a;

    bias ~ normal(0, 10);

    for(d in 1:D){
        W_1[d] ~ normal(0, 10);
        W_2[d] ~ normal(0, 10);
        B[d] ~ normal(0, 10);
    }

    for (n in 1:N){
        a = bias;
        for(d in 1:D){
            a += X[n, d] * W_1[d];
            a += (X[n, d] + B[n]) * (X[n, d] + B[n]) * W_2[d];
        }
        Y[n] ~ bernoulli(inv_logit(normal(a, P / O[n])));
    }
}