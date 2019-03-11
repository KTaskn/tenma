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
    real <lower=0> s[N, H];
}

model {

    W ~ normal(0, 10000.0);
    bias ~ normal(0, 10000.0);

    for (n in 1:N){
        s[n] ~ normal(inv_logit(sum(X[n] * W) + bias), 0.1);
    }

    for (n in 1:N){
        real y = 0.0;
        for (h in 1:H){
            real tmp_2 = 0.0;
            for(hi in 1:h){
                tmp_2 = tmp_2 + exp(s[n, hi]);
            }
            y *= exp(s[n, h]) / tmp_2;
        }
        Y[n] ~ normal(y, 0.1);
    }
}