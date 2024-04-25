functions {

    matrix self_differences(vector input_vec){
        int N = num_elements(input_vec);

        row_vector[N] rv = input_vec';

        matrix[N, N] output_mat = rep_matrix(input_vec, N) - rep_matrix(rv, N);

        return output_mat;
    }

    real log_likelihood(real mu, real alpha, real delta, int N, real max_T, vector differences_from_max_T, matrix exp_differences_mat, row_vector rv) {

        // First
        vector[N] summands = exp(-delta * differences_from_max_T);
        real first = mu * max_T - (alpha / delta) * (sum(summands) - N);

        // Second
        matrix[N, N] inner_sum_mat = pow(exp_differences_mat, delta);
        row_vector[N] term_inside_log = mu + alpha * (rv * inner_sum_mat);
        real second = sum(log(term_inside_log));
        
        return (-first + second);
    }
}

data {
    int<lower=0> N;
    vector[N] events_list;
    real max_T;
}

transformed data {
    vector[N] differences_from_max_T = max_T - events_list;

    // Create matrix of differences between event times
    matrix[N, N] exp_differnces_mat = exp(self_differences(events_list));

    // Zero-out the lower triangle including main diagonal
    for (n in 1:N) {
        for (m in n:N) {
            exp_differnces_mat[m, n] = 0;
        }
    }

    // Define row_vector for computing column sum later
    row_vector[N] rv = rep_row_vector(1.0,N);

}

parameters {
    real<lower=0> mu;
    real<lower=0> alpha;
    real<lower=0> delta;
}

model {
    mu ~ uniform(0, 1e6);
    alpha ~ uniform(0, 1e6);
    delta ~ uniform(0, 1e6);

    target += log_likelihood(mu, alpha, delta, N, max_T, differences_from_max_T, exp_differnces_mat, rv);
}