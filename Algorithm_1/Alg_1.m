    
	% Test the resampling idea with target function f(x) = sinint( v * x / a ) * exp( -|x|^2 / 2 ) or f(x) = exp( -|v * x| / a ) * exp( -|x|^2 / 2 )
	
    clear

    %% Preparation of the data set and implementation details
	use_init_omega_zero = false;    % If set to true, use zero initial values for all frequencies; otherwise, use standard Gaussian distribution for initial frequencies.
	use_random_vector_v = true;    % If set to true, use random uniform distribution in each direction of vector v; otherwise, use only the first component equal to 1, and other components equal to 0.
	use_periodic_data = false;    % If set to true, use periodic data for training; otherwise, use non-periodic original sampled data.
	use_sinint_f = false;    % If set to true, use the target function with f(x) = sinint( x_data_prod_v / a ) * exp( -x_data_square / 2 ); otherwise use the exponential target function.
	use_noisy_y_data = true;    % If set to true, add a Gaussian noise with mean zero and standard deviation 0.25 on each observed y_j data points; otherwise use purely clean data
	use_standard_data = false;    % If set to true, use standardized training data; otherwise use orginal training data.
	use_early_stopping = true;    % If set to true, implement an early stopping by inspecting on the validation loss.

	J = 18750;
	d = 2;
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

	if( use_periodic_data )
		x_transformed_data = mod( x_data + L, 2 * L ) - L;
		x_data_square = sum( x_transformed_data.^2, 2 );
		x_data_prod_v = x_transformed_data * v_vec';
		
		x_transformed_data_test = mod( x_data_test + L, 2 * L ) - L;
		x_data_test_square = sum( x_transformed_data_test.^2, 2 );
		x_data_test_prod_v = x_transformed_data_test * v_vec';
	else
		x_data_square = sum( x_data.^2, 2 );
		x_data_prod_v = x_data * v_vec';
		
		x_data_test_square = sum( x_data_test.^2, 2 );
		x_data_test_prod_v = x_data_test * v_vec';
	end
    
	if( use_sinint_f )   
		y_data_true = sinint( x_data_prod_v / a_para ) .* exp( -x_data_square / 2 );
		y_data_test = sinint( x_data_test_prod_v / a_para ) .* exp( -x_data_test_square / 2 );
	else
		y_data_true = exp( -abs( x_data_prod_v ) / a_para ) .* exp( -x_data_square / 2 );
		y_data_test = exp( -abs( x_data_test_prod_v ) / a_para ) .* exp( -x_data_test_square / 2 );
	end
	
	if( use_noisy_y_data )
		s_para = 0.25 / 16;
		y_data_ori = y_data_true + s_para * randn( J, 1 );
	else
		y_data_ori = y_data_true;
	end
	
	if( use_standard_data )
		mu_y_sample = mean( y_data_ori );
		sigma_y_sample = std( y_data_ori );
		y_data = ( y_data_ori - mu_y_sample ) / sigma_y_sample;
	else
		y_data = y_data_ori;
	end
	
	x_data_train = x_data( 1 : J * ( 1 - valid_train_ratio ), : );
	y_data_train = y_data( 1 : J * ( 1 - valid_train_ratio ), 1 );
	x_data_valid = x_data( J * ( 1 - valid_train_ratio ) + 1 : J, : );
	y_data_valid = y_data( J * ( 1 - valid_train_ratio ) + 1 : J, 1 );
	J_train = ( 1 - valid_train_ratio ) * J;
	y_data_valid_ori = y_data_ori( J_train + 1 : J, : );
    y_data_train_ori = y_data_ori( 1 : J_train, : );													
	
	if( use_early_stopping )
		patience = 30;
		beta_best = [];
		best_loss = inf;
		no_improve_count = 0;
	end
	
	%%  Implementation of the resampling algorithm
	num_resample = 200;
	delta = 0.5;
	lambda = K * sqrt( J_train ) / 100;
	Rel_Tol = 1 * 10^( -3 );
    epsilon = 0;    % Algorithm 1 does not involve the \epsilon cut-off parameter
	
	Least_square_rec_test = zeros( num_resample, 1 );
	Least_square_rec_train = zeros( num_resample, 1 );
	Least_square_rec_valid = zeros( num_resample, 1 );
	num_effect_frequencies = zeros( num_resample, 1 );
	
	for n = 1 : 1 : num_resample
		
		zeta_mat = randn( K, d );
		omega_rw = omega_sample + delta * zeta_mat;
	
		S_mat = exp( 1i * ( x_data_train * omega_rw' ) );
		% B_mat = S_mat' * S_mat + lambda * J * eye( K );
		c_vec = S_mat' * y_data_train;
		
		% Solve the system ( S^T * S + lambda * I ) * beta = S^T * b using pcg    % beta_vec = pcg( B_mat, c_vec, Rel_Tol );
		max_iter = 100;
        [ beta_vec, flag, relres, iter ] = pcg( @( x ) applyRegularizedMatrix( S_mat, lambda, x ), c_vec, Rel_Tol, max_iter );

		% Display conjugate gradient solver results
        fprintf( 'resample_iter_num = %d\n', n );
		if flag == 0
			fprintf( 'PCG converged in %d iterations with relative residual %.2e\n', iter, relres );
		else
			fprintf( 'PCG did not converge. Flag: %d, Relative Residual: %.2e\n', flag, relres );
		end

		S_mat_test = exp( 1i * ( x_data_test * omega_rw' ) );
		S_mat_valid = exp( 1i * ( x_data_valid * omega_rw' ) );
		if( use_standard_data )
			y_test_pred = S_mat_test * beta_vec * sigma_y_sample + mu_y_sample;
			y_train_pred = S_mat * beta_vec * sigma_y_sample + mu_y_sample;
			y_valid_pred = S_mat_valid * beta_vec * sigma_y_sample + mu_y_sample;
		else
			y_test_pred = S_mat_test * beta_vec;
			y_train_pred = S_mat * beta_vec;
			y_valid_pred = S_mat_valid * beta_vec;
		end
		
		% Evaluate the least squares error
		Least_square_rec_test( n, 1 ) = norm( y_test_pred - y_data_test )^2 / norm( y_data_test )^2;
		Least_square_rec_train( n, 1 ) = norm( y_train_pred - y_data_train_ori )^2 / norm( y_data_train_ori )^2;
		Least_square_rec_valid( n, 1 ) = norm( y_valid_pred - y_data_valid_ori )^2 / norm( y_data_valid_ori )^2;
		
		if( use_early_stopping )
			validation_error_n = Least_square_rec_valid( n, 1 );
			if( validation_error_n < best_loss )
				best_loss = validation_error_n;
				beta_best = beta_vec;  % Save current model
				no_improve_count = 0;  % Reset counter
			else
				no_improve_count = no_improve_count + 1;
			end
			
			if( no_improve_count >= patience )
				fprintf( 'Early stopping at resampling iteration %d.\n', n );
				break;
			end
		end
		
		select_indices = abs( beta_vec ) > epsilon;
        beta_vec_select = beta_vec( select_indices, : );
        omega_select = omega_rw( select_indices, : );
		
        weight_vec = abs( beta_vec_select ) / sum( abs( beta_vec_select ) );
        weight_vec_multi = abs( beta_vec ) / sum( abs( beta_vec ) );
		num_effect_frequencies( n, 1 ) = 1 / sum( weight_vec_multi.^2 );    % Compute the effective sample size
        resample_index_set = 1 : length( beta_vec_select );
        resample_indices = randsample( resample_index_set, K, true, weight_vec );
		omega_sample = omega_select( resample_indices, : );
		
	end
	
	if( use_early_stopping )
		if( no_improve_count >= patience )
			Least_square_loss_final = Least_square_rec_test( n, 1 );
			fprintf( 'Early stopping at resampling iteration step %d, least squares generalization error is %6.5e\n', n, Least_square_loss_final );
		else
			Least_square_loss_final = Least_square_rec_test( num_resample, 1 );
			fprintf( 'No early stopping, least squares generalization error in resampling iteration %d is %6.5e\n', num_resample, Least_square_loss_final );
		end
	else
		Least_square_loss_final = Least_square_rec_test( num_resample, 1 );
		fprintf( 'Least squares generalization error in resampling iteration %d is %6.5e\n', num_resample, Least_square_loss_final );
	end

    set( 0, 'DefaultLineLineWidth', 2 );    % Default line width
	set( 0, 'DefaultLineMarkerSize', 6 );  % Default marker size
	set( 0, 'DefaultTextInterpreter', 'latex' );  % LaTeX interpreter for text
	set( 0, 'DefaultAxesTickLabelInterpreter', 'latex' );  % LaTeX for axis ticks
	set( 0, 'DefaultLegendInterpreter', 'latex' );  % LaTeX for legends

	fig_error = figure;
	semilogy( 1 : num_resample, Least_square_rec_test, 'o-' );
	hold on
	semilogy( 1 : num_resample, Least_square_rec_valid, 's-' );
	semilogy( 1 : num_resample, Least_square_rec_train, '*-' );
	legend( 'Test error', 'Validation error', 'Training error' );
	grid on
	
	title_str = sprintf( 'Relative least squares error of random Fourier feature model, $J=%d$, $K=%d$, $\\delta=%.2f$', J_train, K, delta );
	title( title_str );
    xlabel( 'Number of resampling iterations' );
    ylabel( 'Relative least squares error' );

    savefig( fig_error, 'Error_with_n.fig' );
	
	fig_ESS = figure;
	plot( 1 : num_resample, num_effect_frequencies / K, '.' );
    title( 'Ratio between effective frequency sample size $\tilde{K}$ and $K$' )
    xlabel( 'Number of resampling iterations' );
    ylabel( '$\tilde{K}$ / $K$ ' );
	
	if( use_noisy_y_data )														 
		Rel_noise_error = sum( ( y_data_train_ori - y_data_true( 1 : J * ( 1 - valid_train_ratio ), 1 ) ).^2 ) / sum( y_data_train_ori.^2 );
		fprintf( 'Relative squared error in noisy data (Noise-to-signal ratio): %6.5e\n', Rel_noise_error );
	end																

	
	
	
	
    %% Visualization of the distribution of frequencies
    if( ~use_sinint_f && ( d==2 ) )
	    fig_1 = figure;
	    % subplot( 1, 2, 1 );
	    plot( omega_init( :, 1 ), omega_init( :, 2 ), '.' ); % Plot initial omega distribution
	    title( 'Initial frequency distribution' ); 
	    xlabel( '$\omega_1$' );
	    ylabel( '$\omega_2$' );
        savefig( fig_1, 'Omega_dist_initial.fig' );
    
	    %subplot( 1, 2, 2 );
        fig_2 = figure;
	    plot( omega_sample( :, 1 ), omega_sample( :, 2 ), '.' ); % Plot final omega distribution
	    title( 'Final frequency distribution' ); 
	    xlabel( '$\omega_1$' );
	    ylabel( '$\omega_2$' );
	    
	    savefig( fig_2, 'Omega_dist_final.fig' );



        %  Visualize the frequency sample distribution
    
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
        fig_3 = figure;
        num_bins = 100;
	    sample_omega_1 = omega_sample * v_vec';
	    iqr_omega_1 = iqr( sample_omega_1 );
	    % Compute Freedman–Diaconis bin width
	    bin_width_1 = 2 * iqr_omega_1 / K^( 1 / 3 );
	    % Plot histogram using computed bin width
        % histogram( sample_omega_1, 'BinWidth', bin_width_1, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'k' );
        histogram( omega_sample * v_vec', num_bins, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
        hold on
        plot( w_1_values, normalized_conv, 'b-', 'LineWidth', 2 );
        xlabel( '$v\cdot \omega$' );
        ylabel( 'Probability Density' );
		legend( 'Samples of $v\cdot \omega$', 'Optimal probability density' );
        savefig( fig_3, 'rotate_omega_1_dist.fig' );
        
        % Display C
        fprintf('Approximate normalization constant C: %.6f\n', C);
    
        fig_4 = figure;
        num_bins = 100;
	    v_vec_ortho = [ -v_vec( 1, 2), v_vec( 1, 1 ) ];
	    sample_omega_2 = omega_sample * v_vec_ortho';
	    iqr_omega_2 = iqr( sample_omega_2 );
	    % Compute Freedman–Diaconis bin width
	    bin_width_2 = 2 * iqr_omega_2 / K^( 1 / 3 );
	    % Plot histogram using computed bin width
        % histogram( sample_omega_2, 'BinWidth', bin_width_2, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'k' );
        histogram( omega_sample * v_vec_ortho', num_bins, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
        hold on
        omega_2_values = linspace( -10, 10, 200 );
        p2_omega_2_vals = exp( - omega_2_values.^2 / 2 ) / sqrt( 2 * pi );
        plot( omega_2_values, p2_omega_2_vals , 'b-', 'LineWidth', 2 );
        xlabel( '$v^\perp\cdot \omega$' );
        ylabel( 'Probability Density' );
		legend( 'Samples of $v^\perp \cdot \omega$', 'Optimal probability density' );
        savefig( fig_4, 'rotate_omega_2_dist.fig' );

    end
									
											 
		
			 
			  
	
			   
															   
			 
								 
							   
	  
			  
					
					
		   
		  
	   
    %% Subroutine for conjugate gradient method 
    function y = applyRegularizedMatrix( A, lambda, x )
	    % Compute (A^T A + lambda I) * x
	    y = A' * ( A * x ) + lambda * x;
    end
	    
	
	
	
	
	
	
	