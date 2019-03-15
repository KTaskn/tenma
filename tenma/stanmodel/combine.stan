data {
    // data length
    int N;
    int H;
    int D;
    real X[N, H, D];
    real Y[N];
}

parameters {
    real W[D];
    real bias;
}

model {
    real s[H];
    real y;
    real tmp;
    real a;

    bias ~ normal(0, 10000);

    for(d in 1:D){
        W[d] ~ normal(0, 10000);
    }

    for (n in 1:N){
        for (h in 1:H){
            a = bias;
            for(d in 1:D){
                a += X[n, h, d] * W[d];
            }
            s[h] = a;
        }

        y = 1.0;
        for (h in 1:H){
            tmp = 0.0;
            for(hi in h:H){
                tmp += exp(s[hi]);
            }
            y *= exp(s[h]) / tmp;
        }
        Y[n] ~ normal(y, 1);
    }
}