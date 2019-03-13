data {
    // data length
    int N;
    int H;
    int D;
    matrix[H, D] X[N];
    real Y[N];
}

parameters {
    vector[D] W;
    real bias;
    real<lower=0> s[N, H];
}

model {

    W ~ normal(0, 1.0);
    bias ~ normal(0, 0.25);

    for (n in 1:N){
        //s[n] ~ lognormal(exp(bias + sum(X[n] * W)), 0.125);
        s[n] ~ normal(bias + sum(X[n] * W), 0.125);
    }

    for (n in 1:N){
        real y = 1.0;
        for (h in 1:H){
            real tmp = 0.0;
            for(hi in h:H){
                tmp += exp(s[n, hi]);
            }
            y *= exp(s[n, h]) / tmp;
        }
        Y[n] ~ normal(y, 0.01);
    }
}