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
    y ~ normal( mu , sigma );
}