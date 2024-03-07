data{
    int<lower=1> N;
    // real y[N]; // think this is wrong
    // vector[N] y; // but this is a valid option
    array [N] real y;
}
parameters{
    real mu;
    real<lower=0> sigma;
}
model{
    sigma ~ cauchy( 0 , 1 );
    mu ~ normal( 0 , 10 );
    y ~ normal( mu , sigma );
}
