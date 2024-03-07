data{
    int<lower=1> N;
    // real y[N];
    vector[N] y;
}
parameters{
    real alpha1;
    real alpha2;
    real<lower=0> sigma;
}
transformed parameters{
    real mu= alpha1 + alpha2;
}
model{
    sigma ~ cauchy( 0 , 1 );
    alpha1 ~ normal(0, 10);
    alpha2 ~ normal(0, 10);
    y ~ normal( mu , sigma );
}