data {
    // data length
    int N;
    int D;
    real X[N, D];
    int O[N];
    int Y[N];
}

parameters {
    real W[D];
    real bias;
}

model {
    bias ~ normal(0, 10);

    for(d in 1:D){
        W[d] ~ normal(0, 10);
    }

    for (n in 1:N){
        real a = bias;
        for(d in 1:D){
            a += X[n, d] * W[d];
        }
        Y[n] ~ bernoulli(inv_logit(a));
        O[n] ~ poisson(1.0 / inv_logit(a));
    }
}