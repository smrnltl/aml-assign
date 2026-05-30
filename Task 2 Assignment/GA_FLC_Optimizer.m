%% =========================================================================
%  GA_FLC_Optimizer.m
%
%  Genetic Algorithm to optimise membership function parameters of the
%  Mamdani FLC designed in Part 1.
%
%  CHROMOSOME ENCODING:
%    Each MF is parameterised by its defining breakpoints.
%    - trapmf needs 4 params [a b c d]
%    - trimf  needs 3 params [a b c]  (padded to 4 by repeating last)
%
%  Input MF gene allocation (only input MFs are optimised):
%    temperature   : 3 MFs × 4 params = 12 genes
%    humidity      : 3 MFs × 4 params = 12 genes
%    external_light: 3 MFs × 4 params = 12 genes
%    time_of_day   : 4 MFs × 4 params = 16 genes  ← 4 MFs, not 3
%    occupancy     : 3 MFs × 4 params = 12 genes
%    user_activity : 3 MFs × 4 params = 12 genes
%                                     ----------
%  Total chromosome length = 76 real-valued genes
%
%  NOTE on full-system encoding (for Mamdani vs Sugeno comparison):
%    If output MFs were also optimised, 12 output MFs × 4 params = 48 genes
%    would be added, giving 76 + 48 = 124 genes total.
%    A first-order Sugeno model would instead add 25 rules × 3 outputs × 7
%    coefficients = 525 output params, giving 76 + 525 = 601 genes total.
%
%  FITNESS FUNCTION:
%    Root-Mean-Squared-Error (RMSE) between FLC output and a
%    synthetic reference dataset of 200 input-output examples.
%    Fitness = 1 / (1 + RMSE)   [maximise → minimise RMSE]
%
% =========================================================================

clear; clc; close all;

% Suppress FIS warnings during GA optimisation
warning('off', 'fuzzy:general:evalfis_noRuleFired');
warning('off', 'fuzzy:general:evalfis_EmptyOutputFuzzySet');

%% -----------------------------------------------------------------------
%  STEP 1 – Load base FIS (built in Part 1)
% -----------------------------------------------------------------------
% If running standalone, rebuild FIS first
if ~exist('SmartApartmentFLC.fis','file')
    run('SmartApartmentFLC.m');
    close all;
end
baseFIS = readfis('SmartApartmentFLC.fis');

%% -----------------------------------------------------------------------
%  STEP 2 – Generate Synthetic Reference Dataset
%  (200 samples; in a real deployment these are logged sensor readings)
% -----------------------------------------------------------------------
rng(42);   % reproducibility
N_data = 200;
temp_d   = 5  + 35*rand(N_data,1);   % 5–40 °C
hum_d    = 20 + 80*rand(N_data,1);   % 20–100 %
light_d  = 50 + 950*rand(N_data,1);  % 50–1000 lux
tod_d    = 24*rand(N_data,1);         % 0–24 hr
occ_d    = randi([0 5], N_data,1);    % 0–5 persons
act_d    = 10*rand(N_data,1);         % 0–10

X_data = [temp_d, hum_d, light_d, tod_d, occ_d, act_d];

% Reference outputs: simulate ideal targets with small Gaussian noise
Y_ref = zeros(N_data, 3);
for i = 1:N_data
    Y_ref(i,:) = evalfis(baseFIS, X_data(i,:));
end
% Add realistic sensor/reference noise (5% of range)
Y_ref(:,1) = Y_ref(:,1) + 10*randn(N_data,1);   % HVAC  range ±100
Y_ref(:,2) = Y_ref(:,2) +  5*randn(N_data,1);   % Light range 0–100
Y_ref(:,3) = Y_ref(:,3) +  5*randn(N_data,1);   % Blind range 0–100

fprintf('Reference dataset: %d samples generated.\n', N_data);

%% -----------------------------------------------------------------------
%  STEP 3 – Define Chromosome Structure
%  We optimise only the INPUT membership function parameters to keep the
%  chromosome tractable while demonstrating the GA mechanism clearly.
%
%  6 inputs × 3 MFs each:
%    temp    : [trapmf×4, trimf×4, trapmf×4]  = 12 genes
%    humidity: [trapmf×4, trimf×4, trapmf×4]  = 12 genes
%    ext_light:[trapmf×4, trimf×4, trapmf×4]  = 12 genes
%    tod     : [trapmf×4, trimf×4, trimf×4, trapmf×4] = 16 genes (4 MFs)
%    occ     : [trapmf×4, trimf×4, trapmf×4]  = 12 genes
%    activity: [trapmf×4, trimf×4, trapmf×4]  = 12 genes
%
%  Total = 12+12+12+16+12+12 = 76 genes (real-valued)
% -----------------------------------------------------------------------

% Bounds: [lower_bounds ; upper_bounds] for each gene
% Input ranges: temp[0,40] hum[0,100] light[0,1000] tod[0,24] occ[0,5] act[0,10]
ranges = [0,40; 0,100; 0,1000; 0,24; 0,5; 0,10];

% Build lb / ub by repeating input range for all MF params of that input
lb = []; ub = [];
n_mf_genes = [12, 12, 12, 16, 12, 12];  % genes per input
for inp = 1:6
    lo = ranges(inp,1); hi = ranges(inp,2);
    lb = [lb, repmat(lo, 1, n_mf_genes(inp))]; %#ok<AGROW>
    ub = [ub, repmat(hi, 1, n_mf_genes(inp))]; %#ok<AGROW>
end
n_genes = numel(lb);   % = 76
fprintf('Chromosome length: %d genes\n', n_genes);

%% -----------------------------------------------------------------------
%  STEP 4 – GA Parameters
% -----------------------------------------------------------------------
GA_params.pop_size     = 60;      % population size
GA_params.n_generations= 80;      % max generations
GA_params.p_crossover  = 0.80;    % crossover probability
GA_params.p_mutation   = 0.05;    % per-gene mutation probability
GA_params.elite_count  = 2;       % elitism: carry top-2 to next gen
GA_params.tournament_k = 3;       % tournament selection size
GA_params.sigma_init   = 0.1;     % initial mutation step (fraction of range)

fprintf('\nGA Parameters:\n');
fprintf('  Population size  : %d\n', GA_params.pop_size);
fprintf('  Generations      : %d\n', GA_params.n_generations);
fprintf('  Crossover prob   : %.2f\n', GA_params.p_crossover);
fprintf('  Mutation prob    : %.2f\n', GA_params.p_mutation);
fprintf('  Elitism count    : %d\n', GA_params.elite_count);

%% -----------------------------------------------------------------------
%  STEP 5 – Fitness Function
% -----------------------------------------------------------------------
fitness_fn = @(chrom) evaluate_fitness(chrom, baseFIS, X_data, Y_ref, lb, ub);

%% -----------------------------------------------------------------------
%  STEP 6 – Initialise Population
% -----------------------------------------------------------------------
pop = lb + (ub - lb) .* rand(GA_params.pop_size, n_genes);
% Enforce valid MF ordering within each chromosome
pop = enforce_mf_ordering(pop, n_mf_genes, ranges);

fitness = zeros(GA_params.pop_size, 1);
for i = 1:GA_params.pop_size
    fitness(i) = fitness_fn(pop(i,:));
end

best_fitness_history = zeros(GA_params.n_generations, 1);
mean_fitness_history = zeros(GA_params.n_generations, 1);
best_rmse_history    = zeros(GA_params.n_generations, 1);

fprintf('\nRunning GA optimisation...\n');
fprintf('%-10s %-15s %-15s %-15s\n','Generation','Best Fitness','Mean Fitness','Best RMSE');

%% -----------------------------------------------------------------------
%  STEP 7 – Main GA Loop
% -----------------------------------------------------------------------
for gen = 1:GA_params.n_generations

    % --- Elitism: preserve top individuals ---
    [sorted_fit, sort_idx] = sort(fitness, 'descend');
    elite_pop = pop(sort_idx(1:GA_params.elite_count), :);

    new_pop = elite_pop;  % seed new population with elites

    % --- Fill remainder of population ---
    while size(new_pop, 1) < GA_params.pop_size

        % Tournament selection (parent 1)
        p1 = tournament_select(pop, fitness, GA_params.tournament_k);

        % Tournament selection (parent 2)
        p2 = tournament_select(pop, fitness, GA_params.tournament_k);

        % --- Simulated Binary Crossover (SBX) ---
        if rand < GA_params.p_crossover
            eta_c = 2;  % distribution index
            [c1, c2] = sbx_crossover(p1, p2, lb, ub, eta_c);
        else
            c1 = p1; c2 = p2;
        end

        % --- Polynomial Mutation ---
        c1 = poly_mutation(c1, lb, ub, GA_params.p_mutation);
        c2 = poly_mutation(c2, lb, ub, GA_params.p_mutation);

        % --- Enforce valid MF ordering ---
        c1 = enforce_mf_ordering(c1, n_mf_genes, ranges);
        c2 = enforce_mf_ordering(c2, n_mf_genes, ranges);

        new_pop = [new_pop; c1; c2]; %#ok<AGROW>
    end

    % Trim to exact population size
    pop    = new_pop(1:GA_params.pop_size, :);

    % Evaluate fitness
    for i = 1:GA_params.pop_size
        fitness(i) = fitness_fn(pop(i,:));
    end

    % Record stats
    best_fitness_history(gen) = max(fitness);
    mean_fitness_history(gen) = mean(fitness);
    best_rmse_history(gen)    = 1/max(fitness) - 1;  % inverse of fitness

    if mod(gen, 10) == 0 || gen == 1
        fprintf('%-10d %-15.6f %-15.6f %-15.4f\n', gen, ...
            best_fitness_history(gen), mean_fitness_history(gen), ...
            best_rmse_history(gen));
    end
end

%% -----------------------------------------------------------------------
%  STEP 8 – Extract Best Solution
% -----------------------------------------------------------------------
[~, best_idx] = max(fitness);
best_chrom    = pop(best_idx, :);
best_fis      = decode_chromosome(best_chrom, baseFIS, n_mf_genes, ranges);

% Compare RMSE: base FIS vs optimised FIS
rmse_base = compute_rmse(baseFIS, X_data, Y_ref);
rmse_opt  = compute_rmse(best_fis, X_data, Y_ref);

fprintf('\n========================================\n');
fprintf('  OPTIMISATION RESULTS\n');
fprintf('========================================\n');
fprintf('  Base FIS RMSE    : %.4f\n', rmse_base);
fprintf('  Optimised RMSE   : %.4f\n', rmse_opt);
fprintf('  Improvement      : %.2f%%\n', 100*(rmse_base-rmse_opt)/rmse_base);
fprintf('  Chromosome Length: %d genes\n', n_genes);

%% -----------------------------------------------------------------------
%  STEP 9 – Convergence Plots
% -----------------------------------------------------------------------
figure('Name','GA Convergence','NumberTitle','off','Position',[50 50 1000 450]);

subplot(1,2,1);
plot(1:GA_params.n_generations, best_fitness_history, 'b-', 'LineWidth',2); hold on;
plot(1:GA_params.n_generations, mean_fitness_history, 'r--','LineWidth',1.5);
legend('Best Fitness','Mean Fitness','Location','best');
xlabel('Generation'); ylabel('Fitness = 1/(1+RMSE)');
title('GA Convergence – Fitness over Generations');
grid on; xlim([1 GA_params.n_generations]);

subplot(1,2,2);
semilogy(1:GA_params.n_generations, best_rmse_history, 'g-', 'LineWidth',2);
xlabel('Generation'); ylabel('RMSE (log scale)');
title('GA Convergence – Best RMSE over Generations');
grid on; xlim([1 GA_params.n_generations]);

sgtitle('Genetic Algorithm: FLC Membership Function Optimisation','FontSize',13,'FontWeight','bold');

%% -----------------------------------------------------------------------
%  STEP 10 – Compare MFs Before vs After
% -----------------------------------------------------------------------
figure('Name','MF Comparison: Base vs Optimised','NumberTitle','off','Position',[50 550 1400 500]);
inputNames = {'temperature','humidity','external\_light','time\_of\_day','occupancy\_level','user\_activity'};
for k = 1:6
    subplot(2,6,k);
    plotmf(baseFIS,'input',k);
    title(['Base: ' inputNames{k}]);
    xlabel(''); grid on;

    subplot(2,6,k+6);
    plotmf(best_fis,'input',k);
    title(['Opt: ' inputNames{k}]);
    xlabel(''); grid on;
end
sgtitle('Input MFs: Base FIS (top) vs GA-Optimised FIS (bottom)','FontSize',12,'FontWeight','bold');

%% -----------------------------------------------------------------------
%  STEP 11 – Sugeno Comparison Discussion (as required by assignment)
% -----------------------------------------------------------------------
fprintf('\n--- Mamdani vs Sugeno GA Chromosome Discussion ---\n');
fprintf('Current model: Mamdani\n');
fprintf('  Chromosome optimises MF breakpoints (crisp params).\n');
fprintf('  Output MFs are fuzzy sets → defuzzification via centroid.\n');
fprintf('  Chromosome length for full system = %d input genes\n', n_genes);
fprintf('  + output MF genes (48) = %d total genes if outputs included.\n', n_genes+48);
fprintf('\nIf Sugeno (TSK) model were used instead:\n');
fprintf('  Output MFs are replaced by linear/constant functions:\n');
fprintf('    y_k = a_k0 + a_k1*x1 + ... + a_k6*x6  (linear Sugeno)\n');
fprintf('    or y_k = c_k  (constant/zero-order Sugeno)\n');
fprintf('  Each rule has 6 input coefficients + 1 bias = 7 params.\n');
fprintf('  With 25 rules and 3 outputs: 25×3×7 = 525 output params.\n');
fprintf('  Total GA chromosome (Sugeno) ≈ %d + 525 = %d genes.\n', n_genes, n_genes+525);
fprintf('  Advantage: Sugeno outputs are continuous → smoother GA landscape.\n');
fprintf('  Disadvantage: less interpretable output MFs for disabled care context.\n');

writeFIS(best_fis, 'SmartApartmentFLC_optimised');
fprintf('\nOptimised FIS saved to SmartApartmentFLC_optimised.fis\n');

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function fit = evaluate_fitness(chrom, baseFIS, X, Y_ref, lb, ub)
% Suppress expected warnings during GA exploration
    warning('off', 'fuzzy:general:evalfis_outOfRangeInput');
    warning('off', 'fuzzy:general:evalfis_noRuleFired');
    warning('off', 'fuzzy:general:evalfis_EmptyOutputFuzzySet');

    chrom = max(lb, min(ub, chrom));
    n_mf_genes = [12, 12, 12, 16, 12, 12];
    ranges = [0,40; 0,100; 0,1000; 0,24; 0,5; 0,10];
    try
        fis  = decode_chromosome(chrom, baseFIS, n_mf_genes, ranges);
        rmse = compute_rmse(fis, X, Y_ref);
        fit  = 1 / (1 + rmse);
    catch
        fit = 0;
    end

    warning('on', 'fuzzy:general:evalfis_outOfRangeInput');
    warning('on', 'fuzzy:general:evalfis_noRuleFired');
    warning('on', 'fuzzy:general:evalfis_EmptyOutputFuzzySet');
end

function fis = decode_chromosome(chrom, baseFIS, n_mf_genes, ranges)
    % Reconstruct FIS from chromosome by updating MF parameters
    fis   = baseFIS;
    ptr   = 1;   % pointer into chromosome
    n_in  = 6;
    mf_per_input = [3, 3, 3, 4, 3, 3];  % MFs per input

    for inp = 1:n_in
        genes = chrom(ptr : ptr + n_mf_genes(inp) - 1);
        ptr   = ptr + n_mf_genes(inp);
        genes = sort(genes);  % enforce monotonicity

        lo = ranges(inp,1);
        hi = ranges(inp,2);
        genes = max(lo, min(hi, genes));  % clamp within input range

        gene_ptr = 1;
        for mf_idx = 1:mf_per_input(inp)
            params = genes(gene_ptr : gene_ptr+3);
            gene_ptr = gene_ptr + 4;
            params = sort(params);
            % Determine type from base FIS and extract the correct number of
            % parameters. trimf uses 3 params [a b c]; we store 4 genes per
            % MF slot for uniform chromosome length, so we use positions
            % [1, 2, 4] — taking the lowest, middle, and highest sorted
            % values as left-base, peak, and right-base respectively.
            % This preserves a valid triangle (a ≤ b ≤ c) after sorting.
            % trapmf uses all 4 params [a b c d] directly.
            mf_type = baseFIS.Inputs(inp).MembershipFunctions(mf_idx).Type;
            if strcmp(mf_type, 'trimf')
                params_use = [params(1), params(2), params(4)];
            else
                params_use = params;  % trapmf: all 4 sorted params
            end
            fis.Inputs(inp).MembershipFunctions(mf_idx).Parameters = params_use;
        end
    end
end

function rmse = compute_rmse(fis, X, Y_ref)
    N = size(X, 1);
    Y_pred = zeros(N, 3);
    for i = 1:N
        try
            Y_pred(i,:) = evalfis(fis, X(i,:));
        catch
            Y_pred(i,:) = 0;
        end
    end
    rmse = sqrt(mean((Y_pred - Y_ref).^2, 'all'));
end

function parent = tournament_select(pop, fitness, k)
    n = size(pop,1);
    idx = randperm(n, k);
    [~, best] = max(fitness(idx));
    parent = pop(idx(best), :);
end

function [c1, c2] = sbx_crossover(p1, p2, lb, ub, eta)
    n  = numel(p1);
    c1 = p1; c2 = p2;
    for j = 1:n
        if rand < 0.5
            if abs(p1(j)-p2(j)) > 1e-10
                y1 = min(p1(j),p2(j));
                y2 = max(p1(j),p2(j));
                r  = rand;
                beta1 = 1 + 2*(y1-lb(j))/(y2-y1);
                beta2 = 1 + 2*(ub(j)-y2)/(y2-y1);
                alpha1 = 2 - beta1^(-(eta+1));
                alpha2 = 2 - beta2^(-(eta+1));
                if r <= 1/alpha1
                    betaq1 = (r*alpha1)^(1/(eta+1));
                else
                    betaq1 = (1/(2-r*alpha1))^(1/(eta+1));
                end
                if r <= 1/alpha2
                    betaq2 = (r*alpha2)^(1/(eta+1));
                else
                    betaq2 = (1/(2-r*alpha2))^(1/(eta+1));
                end
                c1(j) = 0.5*((y1+y2) - betaq1*(y2-y1));
                c2(j) = 0.5*((y1+y2) + betaq2*(y2-y1));
                c1(j) = max(lb(j), min(ub(j), c1(j)));
                c2(j) = max(lb(j), min(ub(j), c2(j)));
            end
        end
    end
end

function child = poly_mutation(chrom, lb, ub, p_mut)
    child  = chrom;
    eta_m  = 20;  % distribution index for mutation
    n      = numel(chrom);
    for j = 1:n
        if rand < p_mut
            delta1 = (chrom(j) - lb(j)) / (ub(j) - lb(j));
            delta2 = (ub(j) - chrom(j)) / (ub(j) - lb(j));
            r = rand;
            if r < 0.5
                xy     = 1 - delta1;
                val    = 2*r + (1-2*r)*xy^(eta_m+1);
                deltaq = val^(1/(eta_m+1)) - 1;
            else
                xy     = 1 - delta2;
                val    = 2*(1-r) + 2*(r-0.5)*xy^(eta_m+1);
                deltaq = 1 - val^(1/(eta_m+1));
            end
            child(j) = chrom(j) + deltaq*(ub(j)-lb(j));
            child(j) = max(lb(j), min(ub(j), child(j)));
        end
    end
end

function pop = enforce_mf_ordering(pop, n_mf_genes, ranges)
    % Sort MF params within each input's gene block for validity
    if isvector(pop) && ~ismatrix(pop)
        pop = pop(:)';
    end
    for row = 1:size(pop,1)
        ptr = 1;
        for inp = 1:6
            block = pop(row, ptr:ptr+n_mf_genes(inp)-1);
            lo    = ranges(inp,1);
            hi    = ranges(inp,2);
            block = max(lo, min(hi, sort(block)));
            pop(row, ptr:ptr+n_mf_genes(inp)-1) = block;
            ptr   = ptr + n_mf_genes(inp);
        end
    end
end
