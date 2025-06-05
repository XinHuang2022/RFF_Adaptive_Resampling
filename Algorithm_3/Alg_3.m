    
	% Test the resampling idea with target function f(x) = sinint( v * x / a ) * exp( -|x|^2 / 2 ), without noise in data
	
    clear

    %% Preparation of the data set and implementation details
	use_init_omega_zero = true;    % If set to true, use zero initial values for all frequencies; otherwise, use standard Gaussian distribution for initial frequencies.
	use_random_vector_v = false;    % If set to true, use random uniform distribution in each direction of vector v; otherwise, use only the first component equal to 1, and other components equal to 0.
	use_periodic_data = true;    % If set to true, use periodic data for training; otherwise, use non-periodic original sampled data.
	use_sinint_f = true;    % If set to true, use the target function with f(x) = sinint( x_data_prod_v / a ) * exp( -x_data_square / 2 ); otherwise use the exponential target function.
	use_noisy_y_data = false;    % If set to true, add a Gaussian noise with mean zero and standard deviation 0.25 on each observed y_j data points; otherwise use purely clean data
	

    % seed_val = 2357;
    % seed_val = 2356;
    % rng( seed_val );

	J = 10000;
	d = 4;
	K = 2500;
	
	% Determine the initial distribution of frequency parameters
	if( use_init_omega_zero )
		omega_init = zeros( K, d );
	else
		omega_init = randn( K, d );
	end
	omega_sample = omega_init;
	
	% Preparing the data set ( x_j, y_j ) of size J by d and J by 1
	% Determine the anisotropic property parameter v
	if( use_random_vector_v )
		v_vec = rand( 1, d );
		v_vec = v_vec / norm( v_vec );
		% v_vec = [ 0.3308, 0.9437 ];
	else
		v_vec = zeros( 1, d );
		v_vec( 1, 1 ) = 1;
	end
	a_para = 0.1;
    L = 6;    % Periodicity parameter
	
	valid_train_ratio = 0.2;
	J_train = ( 1 - valid_train_ratio ) * J;
	x_data = randn( J, d );
	test_train_ratio = 0.5;
	x_data_test = randn( test_train_ratio * J_train, d );
	% y_data = zeros( J, 1 );
	if( use_periodic_data )
		x_transformed_data = mod( x_data + L / 2, L ) - L / 2;
		x_data_square = sum( x_transformed_data.^2, 2 );
		x_data_prod_v = x_transformed_data * v_vec';
		
		x_transformed_data_test = mod( x_data_test + L / 2, L ) - L / 2;
		x_data_test_square = sum( x_transformed_data_test.^2, 2 );
		x_data_test_prod_v = x_transformed_data_test * v_vec';
	else
		x_data_square = sum( x_data.^2, 2 );
		x_data_prod_v = x_data * v_vec';
		
		x_data_test_square = sum( x_data_test.^2, 2 );
		x_data_test_prod_v = x_data_test * v_vec';
	end
    
	if( use_sinint_f )   
		y_data = sinint( x_data_prod_v / a_para ) .* exp( -x_data_square / 2 );
		y_data_test = sinint( x_data_test_prod_v / a_para ) .* exp( -x_data_test_square / 2 );
	else
		y_data = exp( -abs( x_data_prod_v ) / a_para ) .* exp( -x_data_square / 2 );
		y_data_test = exp( -abs( x_data_test_prod_v ) / a_para ) .* exp( -x_data_test_square / 2 );
	end
	
	if( use_noisy_y_data )
		xi_noise_level = 0.25;
		y_data = y_data + xi_noise_level * randn( J, 1 );
	end
	
	x_data_train = x_data( 1 : J * ( 1 - valid_train_ratio ), : );
	y_data_train = y_data( 1 : J * ( 1 - valid_train_ratio ), 1 );
	x_data_valid = x_data( J * ( 1 - valid_train_ratio ) + 1 : J, : );
	y_data_valid = y_data( J * ( 1 - valid_train_ratio ) + 1 : J, 1 );
	J_train = ( 1 - valid_train_ratio ) * J;
	
	
	%%  Implementation of the resampling algorithm
	num_resample = 50;
	delta = 0.2;
	lambda = K * sqrt( J_train ) / 100;
	Rel_Tol = 1 * 10^( -3 );
    epsilon = K^(-0.5) / 200;
    % epsilon = 0;
	C_mat_init = eye( d );
	C_mat_sample = C_mat_init;
	epsilon_hat = 0;
	
	Least_square_rec_test = zeros( num_resample, 1 );
	Least_square_rec_train = zeros( num_resample, 1 );
	Least_square_rec_valid = zeros( num_resample, 1 );
	num_effect_frequencies = zeros( num_resample, 4 );    % The first column stores the unique frequency values \tilde{K}, and the second column stores the effective sample size of resampling weight vectors
	
	for n = 1 : 1 : num_resample
	
		L_C_mat = chol( C_mat_sample + epsilon_hat * eye( d ), 'lower' );    % Applying Cholesky decomposition on the covariance matrix
		zeta_mat = randn( K, d ) * L_C_mat;
		omega_rw = omega_sample + delta * zeta_mat;
		%{
		% Define lattice grid spacing
		grid_spacing = pi / L;
		% Project points onto the lattice ( \pi / L ) * Z^d
		omega_new = round( omega_rw / grid_spacing ) * grid_spacing;
		% Find unique frequencies (rows)
		[ omega_unique, ~, ~ ] = unique( omega_new, 'rows' );
		% Output
		K_tilde = size( omega_unique, 1 );  % Number of unique frequencies
		num_effect_frequencies( n, 1 ) = K_tilde;
		fprintf( 'Resampling iteration %d, number of unique frequencies = %d\n', n, K_tilde );
		%}
		
		S_mat = exp( 1i * ( x_data_train * omega_rw' ) );
		% B_mat = S_mat' * S_mat + lambda * J * eye( K );
		c_vec = S_mat' * y_data_train;
		
		% Solve the system ( S^T * S + lambda * I ) * beta = S^T * b using pcg    % beta_vec = pcg( B_mat, c_vec, Rel_Tol );
		max_iter = 100;
        [ beta_vec, flag, relres, iter ] = pcg( @( x ) applyRegularizedMatrix( S_mat, lambda, x ), c_vec, Rel_Tol, max_iter );

		% Display pcg results
        fprintf( 'resample_iter_num = %d\n', n );
		if flag == 0
			fprintf( 'PCG converged in %d iterations with relative residual %.2e\n', iter, relres );
		else
			fprintf( 'PCG did not converge. Flag: %d, Relative Residual: %.2e\n', flag, relres );
		end
		
        % Evaluate the least-squares loss using the resampled frequencies
		select_indices = abs( beta_vec ) > epsilon;
        beta_vec_select = beta_vec( select_indices, : );
		omega_select = omega_rw( select_indices, : );
        weight_vec = abs( beta_vec_select ) / sum( abs( beta_vec_select ) );
		num_effect_frequencies( n, 3 ) = length( beta_vec_select );
        resample_index_set = 1 : length( beta_vec_select );
        resample_indices = randsample( resample_index_set, K, true, weight_vec );
		omega_sample = omega_select( resample_indices, : );
		% select_indices = abs( beta_vec ) > epsilon;
        % beta_vec_select = beta_vec( select_indices, : );
        % omega_select = omega_new( select_indices, : );
        % weight_vec = abs( beta_vec_select ) / sum( abs( beta_vec_select ) );
		% cumulative_weights = cumsum( weight_vec );
		% for k = 1 : 1 : K
			% random_unif = rand;
			% resample_index = find( random_unif <= cumulative_weights, 1 );

			% omega_sample( k, : ) = omega_select( resample_index, : );
        % end

        [ omega_resample_unique, ~, ~ ] = unique( omega_sample, 'rows' );
		
		S_mat_re = exp( 1i * ( x_data_train * omega_resample_unique' ) );
		% B_mat = S_mat' * S_mat + lambda * J * eye( K );
		c_vec_re = S_mat_re' * y_data_train;
		% Solve the system ( S^T * S + lambda * I ) * beta = S^T * b using pcg    % beta_vec = pcg( B_mat, c_vec, Rel_Tol );
		max_iter = 100;
        [ beta_vec_re, flag, relres, iter ] = pcg( @( x ) applyRegularizedMatrix( S_mat_re, lambda, x ), c_vec_re, Rel_Tol, max_iter );
		
		% Evaluate the effective sample size based on the beta_vec_resample weight vector
		weight_vec_resample = abs( beta_vec_re ) / sum( abs( beta_vec_re ) );
		num_effect_frequencies( n, 2 ) = 1 / sum( weight_vec_resample.^2 );
		num_effect_frequencies( n, 4 ) = length( beta_vec_re );
		
		% Compute the testing loss and validation loss
		S_mat_test = exp( 1i * ( x_data_test * omega_resample_unique' ) );
		S_mat_valid = exp( 1i * ( x_data_valid * omega_resample_unique' ) );
		
        Least_square_rec_test( n, 1 ) = norm( S_mat_test * beta_vec_re - y_data_test )^2 / norm( y_data_test )^2;
		Least_square_rec_train( n, 1 ) = norm( S_mat_re * beta_vec_re - y_data_train )^2 / norm( y_data_train )^2;
		Least_square_rec_valid( n, 1 ) = norm( S_mat_valid * beta_vec_re - y_data_valid )^2 / norm( y_data_valid )^2;
		
		omega_bar = mean( omega_sample, 1 );
		omega_center = omega_sample - omega_bar;
		C_hat_n = omega_center' * omega_center / K;
		
		C_mat_sample = ( n * C_mat_sample + C_hat_n ) / ( n + 1 );
		
	end
	
	L_C_mat_final = chol( C_mat_sample + epsilon_hat * eye( d ), 'lower' );
	zeta_mat_final = randn( K, d ) * L_C_mat_final;
	omega_rw_final = omega_sample + delta * zeta_mat_final;
	S_final = exp( 1i * ( x_data_train * omega_rw_final' ) );
	c_final = S_final' * y_data_train;
    [ beta_final, flag, relres, iter ] = pcg( @( x ) applyRegularizedMatrix( S_final, lambda, x ), c_final, Rel_Tol, max_iter );
	if flag == 0
		fprintf( 'PCG converged in %d iterations with relative residual %.2e\n', iter, relres );
	else
		fprintf( 'PCG did not converge. Flag: %d, Relative Residual: %.2e\n', flag, relres );
	end
	
	S_test_final = exp( 1i * ( x_data_test * omega_rw_final' ) );
	Least_square_loss_final = norm( S_test_final * beta_final - y_data_test )^2 / norm( y_data_test )^2;
    Least_square_loss_final
	

    set( 0, 'DefaultLineLineWidth', 2 );    % Default line width
	set( 0, 'DefaultLineMarkerSize', 6 );  % Default marker size
	set( 0, 'DefaultTextInterpreter', 'latex' );  % LaTeX interpreter for text
	set( 0, 'DefaultAxesTickLabelInterpreter', 'latex' );  % LaTeX for axis ticks
	set( 0, 'DefaultLegendInterpreter', 'latex' );  % LaTeX for legends

	figure;
	semilogy( 1 : num_resample, Least_square_rec_test, 'o-' );
	hold on
	semilogy( 1 : num_resample, Least_square_rec_valid, 's-' );
	semilogy( 1 : num_resample, Least_square_rec_train, '*-' );
	legend( 'Testing error', 'Validation error', 'Training error' );
	grid on

    title( 'Relative least squares error for Fourier series, $J=2000$, $K=2500$, $\delta=0.5$' );
    xlabel( 'Number of resampling iteration' );
    ylabel( 'Relative generalization error' );

    savefig( 'Error_with_n.fig' );
	
	figure
	plot( 1 : num_resample, num_effect_frequencies( :, 2 ) ./ num_effect_frequencies( :, 4 ) );
    % plot( 1 : num_resample, num_effect_frequencies( :, 2 ) / K );
    title( 'Relative effective sample size' )
    xlabel( 'Number of resampling iteration' );
    ylabel( 'Effective sample size' );
	
    %% Visualization of the distribution of frequencies
	%{
	figure;
	subplot( 1, 2, 1 );
	plot( omega_init( :, 1 ), omega_init( :, 2 ), '.' ); % Plot initial omega distribution
	title( 'Initial frequency distribution' ); 
	xlabel( 'omega_1' );
	ylabel( 'omega_2' );

	subplot( 1, 2, 2 );
	plot( omega_sample( :, 1 ), omega_sample( :, 2 ), '.' ); % Plot final omega distribution
	title( 'Final frequency distribution' ); 
	xlabel( 'omega_1' );
	ylabel( 'omega_2' );
	
	savefig( 'Omega_dist.fig' );



    %%  Visualize the frequency sample distribution
    % Define parameter a

    % Define the functions
    h = @( x ) exp( -x.^2 / 2 );
    g = @( x ) ( 2 * a_para ) ./ ( 1 + a_para^2 * x.^2 );
    
    % Define convolution function
    convolution = @( x ) integral(@( y ) h( y ) .* g( x - y ), -100, 100, 'ArrayValued', true );
    
    % Compute normalization constant C (integral of convolution from -100 to 100)
    C = integral( @( x ) convolution( x ), -100, 100, 'ArrayValued', true );
    
    % Compute and plot the normalized probability density function
    w_1_values = linspace( -100, 100, 200 ); % Discretized x values
    conv_values = arrayfun( convolution, w_1_values ); % Compute convolution at each x
    normalized_conv = conv_values / C; % Normalize
    
    % Plot the result
    figure;
    num_bins = 100;
    histogram( omega_sample * v_vec', num_bins, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    hold on
    plot( w_1_values, normalized_conv, 'b-', 'LineWidth', 2 );
    xlabel( '$\nu\cdot \omega$' );
    ylabel( 'Probability Density' );

    
    % Display C
    fprintf('Approximate normalization constant C: %.6f\n', C);

    figure;
    num_bins = 100;
    v_vec_ortho = [ -v_vec( 1, 2), v_vec( 1, 1 ) ];
    histogram( omega_sample * v_vec_ortho', num_bins, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    hold on
    omega_2_values = linspace( -10, 10, 200 );
    p2_omega_2_vals = exp( - omega_2_values.^2 / 2 ) / sqrt( 2 * pi );
    plot( omega_2_values, p2_omega_2_vals , 'b-', 'LineWidth', 2 );
    xlabel( '$\nu^\perp\cdot \omega$' );
    ylabel( 'Probability Density' );
    %}
	   
    %% Subroutine for conjugate gradient method 
    function y = applyRegularizedMatrix( A, lambda, x )
	    % Compute (A^T A + lambda I) * x
	    y = A' * ( A * x ) + lambda * x;
    end
	    
	
	
	
	
	
	
	