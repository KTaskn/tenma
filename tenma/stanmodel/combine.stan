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
    // real<lower=0> s[N, H];
}

model {

    W ~ normal(0, 1.5);
    bias ~ normal(0, 1.5);

    /*
    for (n in 1:N){
        // s[n] ~ lognormal(exp(bias + X[n] * W), 0.125);
        s[n] ~ normal(bias + X[n] * W, 0.125);
    }
    */

    for (n in 1:N){
        vector[H] s = bias + X[n] * W;
        real y = 1.0;
        for (h in 1:H){
            real tmp = 0.0;
            for(hi in h:H){
                tmp += exp(s[hi]);
            }
            y *= exp(s[h]) / tmp;
        }
        Y[n] ~ normal(y, 0.01);
    }
}