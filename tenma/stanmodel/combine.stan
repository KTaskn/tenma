data {
    // data length
    int N;
    int D;
    real X[N, D];
    int Y[N];
    real O[N];
    real P;
}

parameters {
    real W_1[D];
    real W_2[D];
    real B[D];
    real s;
    real bias;
}

model {
    bias ~ normal(0, 10);

    for(d in 1:D){
        W_1[d] ~ normal(0, 10);
        W_2[d] ~ normal(0, 10);
        B[d] ~ normal(0, 10);
    }

    for (n in 1:N){
        real a = bias;
        for(d in 1:D){
            a += X[n, d] * W_1[d];
            a += (X[n, d] + B[d]) * (X[n, d] + B[d]) * W_2[d];
        }
        s ~  normal(a, log(P) / log(O[n]));
        Y[n] ~ bernoulli(inv_logit(s));
    }
}