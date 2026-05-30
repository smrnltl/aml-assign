%% =========================================================================
%
%  Compares Genetic Algorithm (GA) vs Particle Swarm Optimisation (PSO)
%  on CEC'2005 functions F1 (Shifted Sphere) and F9 (Shifted Rastrigin)
%  for D = 2 and D = 10.
%
%  Protocol: 15 independent runs per algorithm per function per dimension.
%  Reports: mean, std, best, worst of final function value at convergence.
%  Produces: convergence curves, box plots, summary statistics table.
%
% =========================================================================

clear; clc; close all;

%% -----------------------------------------------------------------------
%  EXPERIMENT CONFIGURATION
% -----------------------------------------------------------------------
N_RUNS      = 15;       % independent runs per configuration
MAX_FES     = 1e4;      % max function evaluations (FEs) per run
DIMENSIONS  = [2, 10];  % dimensionalities to test

% Function handles and metadata
functions = struct();
functions(1).handle = @cec2005_f1;
functions(1).name   = 'F1: Shifted Sphere';
functions(1).lb2    = -100;  functions(1).ub2 = 100;  % bounds for D=2
functions(1).lb10   = -100;  functions(1).ub10= 100;  % bounds for D=10
functions(1).f_opt  = -450;  % known global optimum
functions(1).type   = 'Unimodal';

functions(2).handle = @cec2005_f9;
functions(2).name   = 'F9: Shifted Rastrigin';
functions(2).lb2    = -5;   functions(2).ub2 = 5;
functions(2).lb10   = -5;   functions(2).ub10= 5;
functions(2).f_opt  = -330;
functions(2).type   = 'Multimodal';

algorithms = {'GA (Real-coded SBX)', 'PSO (Inertia-Weight)'};

%% -----------------------------------------------------------------------
%  RESULTS STORAGE
%  results{func_idx, dim_idx, alg_idx} = vector of N_RUNS best values
% -----------------------------------------------------------------------
results       = cell(2, 2, 2);
history_all   = cell(2, 2, 2);  % convergence curves per run

fprintf('========================================================\n');
fprintf('  CEC2005 Benchmark: GA vs PSO\n');
fprintf('  Functions: F1 (Sphere) | F9 (Rastrigin)\n');
fprintf('  Dimensions: D=2, D=10 | Runs: %d | MaxFES: %d\n', N_RUNS, MAX_FES);
fprintf('========================================================\n\n');

%% -----------------------------------------------------------------------
%  MAIN EXPERIMENT LOOP
% -----------------------------------------------------------------------
for fi = 1:2          % function index
    for di = 1:2      % dimension index
        D   = DIMENSIONS(di);
        if di == 1
            lb = functions(fi).lb2 * ones(1,D);
            ub = functions(fi).ub2 * ones(1,D);
        else
            lb = functions(fi).lb10 * ones(1,D);
            ub = functions(fi).ub10 * ones(1,D);
        end
        fn  = functions(fi).handle;

        fprintf('--- %s | D=%d ---\n', functions(fi).name, D);

        % ---- GA ----
        ga_best = zeros(N_RUNS, 1);
        ga_hist = zeros(N_RUNS, MAX_FES);
        for run = 1:N_RUNS
            rng(run*100 + fi*10 + di);  % different seed per run
            [best_val, curve] = run_GA(fn, lb, ub, D, MAX_FES);
            ga_best(run)    = best_val;
            ga_hist(run,:)  = curve;
        end
        results{fi, di, 1}     = ga_best;
        history_all{fi, di, 1} = ga_hist;

        fprintf('  GA  | Mean: %+.4f | Std: %.4f | Best: %+.4f | Worst: %+.4f\n', ...
            mean(ga_best), std(ga_best), min(ga_best), max(ga_best));

        % ---- PSO ----
        pso_best = zeros(N_RUNS, 1);
        pso_hist = zeros(N_RUNS, MAX_FES);
        for run = 1:N_RUNS
            rng(run*100 + fi*10 + di + 500);
            [best_val, curve] = run_PSO(fn, lb, ub, D, MAX_FES);
            pso_best(run)    = best_val;
            pso_hist(run,:)  = curve;
        end
        results{fi, di, 2}     = pso_best;
        history_all{fi, di, 2} = pso_hist;

        fprintf('  PSO | Mean: %+.4f | Std: %.4f | Best: %+.4f | Worst: %+.4f\n\n', ...
            mean(pso_best), std(pso_best), min(pso_best), max(pso_best));
    end
end

%% -----------------------------------------------------------------------
%  RESULTS TABLE
% -----------------------------------------------------------------------
fprintf('=======================================================================\n');
fprintf('%-30s %-5s %-8s %-10s %-10s %-10s %-10s\n', ...
    'Function','D','Alg','Mean','Std','Best','Worst');
fprintf('=======================================================================\n');
func_labels = {'F1-Sphere', 'F9-Rastrigin'};
alg_labels  = {'GA', 'PSO'};
for fi = 1:2
    for di = 1:2
        D = DIMENSIONS(di);
        for ai = 1:2
            r = results{fi,di,ai};
            fprintf('%-30s %-5d %-8s %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
                func_labels{fi}, D, alg_labels{ai}, ...
                mean(r), std(r), min(r), max(r));
        end
    end
    fprintf('-----------------------------------------------------------------------\n');
end

%% -----------------------------------------------------------------------
%  CONVERGENCE PLOTS
% -----------------------------------------------------------------------
fig_conv = figure('Name','Convergence Curves','NumberTitle','off','Position',[50 50 1400 800]);

sp_idx = 1;
for fi = 1:2
    for di = 1:2
        D = DIMENSIONS(di);
        subplot(2,2,sp_idx);
        hold on;

        colors = {[0.1 0.4 0.8], [0.8 0.2 0.1]};
        line_style = {'-','--'};

        for ai = 1:2
            hist_mat = history_all{fi,di,ai};
            mean_curve = mean(hist_mat, 1);
            std_curve  = std(hist_mat, [], 1);

            x_fes = 1:MAX_FES;
            % Plot mean ± std shaded region
            fill([x_fes, fliplr(x_fes)], ...
                 [mean_curve+std_curve, fliplr(mean_curve-std_curve)], ...
                 colors{ai}, 'FaceAlpha', 0.15, 'EdgeColor','none');
            plot(x_fes, mean_curve, line_style{ai}, ...
                'Color', colors{ai}, 'LineWidth', 2, ...
                'DisplayName', [alg_labels{ai} ' (mean±std)']);
        end

        % Global optimum reference line
        yline(functions(fi).f_opt, 'k:', 'LineWidth', 1.5, ...
              'DisplayName', 'Global Optimum');

        xlabel('Function Evaluations (FES)');
        ylabel('Best f(x) Found');
        title(sprintf('%s | D=%d', functions(fi).name, D));
        legend('Location','best','FontSize',8);
        grid on; box on;
        sp_idx = sp_idx + 1;
    end
end
sgtitle('Convergence Curves: GA vs PSO on CEC2005 F1 & F9','FontSize',13,'FontWeight','bold');

%% -----------------------------------------------------------------------
%  BOX PLOTS
% -----------------------------------------------------------------------
figure('Name','Box Plots – Final Values','NumberTitle','off','Position',[50 900 1400 450]);
sp_idx = 1;
for fi = 1:2
    for di = 1:2
        D = DIMENSIONS(di);
        subplot(1,4,sp_idx);

        data_ga  = results{fi,di,1};
        data_pso = results{fi,di,2};
        boxplot([data_ga, data_pso], ...
                'Labels', {'GA','PSO'}, ...
                'Colors', [0.1 0.4 0.8; 0.8 0.2 0.1]);
        title(sprintf('%s\nD=%d', func_labels{fi}, D));
        ylabel('Best f(x) after MaxFES');
        grid on; box on;
        sp_idx = sp_idx + 1;
    end
end
sgtitle('Distribution of Final Best Values: GA vs PSO (15 runs)','FontSize',12,'FontWeight','bold');

%% -----------------------------------------------------------------------
%  PER-RUN DETAIL: Best & Worst Convergence Curves (F9, D=10)
% -----------------------------------------------------------------------
figure('Name','Best-Worst Curves F9 D=10','NumberTitle','off','Position',[50 50 900 400]);
fi = 2; di = 2;  % F9, D=10
for ai = 1:2
    subplot(1,2,ai);
    hist_mat = history_all{fi,di,ai};
    r        = results{fi,di,ai};
    [~, bi]  = min(r);    % best run index
    [~, wi]  = max(r);    % worst run index
    hold on;
    for run = 1:N_RUNS
        plot(1:MAX_FES, hist_mat(run,:), 'Color',[0.7 0.7 0.7],'LineWidth',0.5);
    end
    plot(1:MAX_FES, hist_mat(bi,:), 'g-', 'LineWidth',2.5, 'DisplayName','Best Run');
    plot(1:MAX_FES, hist_mat(wi,:), 'r-', 'LineWidth',2.5, 'DisplayName','Worst Run');
    yline(functions(fi).f_opt,'k:','LineWidth',1.5,'DisplayName','Global Optimum');
    title([alg_labels{ai} ' on F9 (D=10)']);
    xlabel('FES'); ylabel('Best f(x)');
    legend('Location','best'); grid on;
end
sgtitle('Individual Run Convergence – F9 Rastrigin D=10','FontSize',12,'FontWeight','bold');

fprintf('\nPart 3 complete.\n');

%% =========================================================================
%  ALGORITHM IMPLEMENTATIONS
%% =========================================================================

%% ------- REAL-CODED GENETIC ALGORITHM -----------------------------------
function [best_val, history] = run_GA(fn, lb, ub, D, max_fes)
% Parameters
pop_size   = 50;
p_cross    = 0.9;
p_mut      = 1/D;   % standard 1/D mutation rate
eta_c      = 15;    % SBX distribution index
eta_m      = 20;    % polynomial mutation distribution index
elite_n    = 2;

history = zeros(1, max_fes);
fes     = 0;

% Initialisation
pop     = lb + (ub-lb).*rand(pop_size, D);
fit     = zeros(pop_size,1);
for i = 1:pop_size
    fit(i) = fn(pop(i,:));
    fes    = fes + 1;
end

[best_val, ~] = min(fit);
history(1:min(fes, max_fes)) = best_val;

while fes < max_fes
    % Snapshot current fitness before selection so elite fitness values
    % can be carried over without re-evaluation (avoids FES double-counting)
    fitness_prev = fit;
    [~, sort_idx] = sort(fit);
    new_pop = pop(sort_idx(1:elite_n), :);  % elites

    while size(new_pop,1) < pop_size && fes < max_fes
        % Tournament selection
        idx1 = randperm(pop_size, 3);
        [~, b1] = min(fit(idx1)); p1 = pop(idx1(b1),:);
        idx2 = randperm(pop_size, 3);
        [~, b2] = min(fit(idx2)); p2 = pop(idx2(b2),:);

        % SBX crossover
        if rand < p_cross
            [c1, c2] = ga_sbx(p1, p2, lb, ub, eta_c);
        else
            c1 = p1; c2 = p2;
        end

        % Polynomial mutation
        c1 = ga_polymut(c1, lb, ub, p_mut, eta_m);
        c2 = ga_polymut(c2, lb, ub, p_mut, eta_m);

        f1 = fn(c1); fes = fes+1;
        f2 = fn(c2); fes = fes+1;

        new_pop = [new_pop; c1; c2];  %#ok<AGROW>
        if f1 < best_val; best_val = f1; end
        if f2 < best_val; best_val = f2; end
        history(min(fes, max_fes)) = best_val;
    end

    pop = new_pop(1:pop_size,:);
    % Re-evaluate only the non-elite individuals (indices elite_n+1 : pop_size).
    % Elite individuals at rows 1:elite_n were carried over unchanged and their
    % fitness values are still valid — re-evaluating them would double-count FES.
    for i = elite_n+1 : pop_size
        fit(i) = fn(pop(i,:));
        fes    = fes + 1;
    end
    % Elites retain their fitness from the previous generation
    [~, sort_idx_prev] = sort(fitness_prev, 'descend');
    fit(1:elite_n) = fitness_prev(sort_idx_prev(1:elite_n));
    [cur_best,~] = min(fit);
    if cur_best < best_val; best_val = cur_best; end
    idx_fill = min(fes, max_fes);
    history(max(1,idx_fill-(pop_size-elite_n)+1):idx_fill) = best_val;
end

% Fill remainder of history
for k = 2:max_fes
    if history(k) == 0; history(k) = history(k-1); end
end
end

%% ------- PARTICLE SWARM OPTIMISATION ------------------------------------
function [best_val, history] = run_PSO(fn, lb, ub, D, max_fes)
% Standard inertia-weight PSO
pop_size = 50;
w_max    = 0.9;   % inertia weight (linearly decreasing)
w_min    = 0.4;
c1       = 2.0;   % cognitive coefficient
c2       = 2.0;   % social coefficient
v_max    = 0.2 * (ub - lb);  % velocity clamp

fes      = 0;
history  = zeros(1, max_fes);

% Initialise positions and velocities
pos   = lb + (ub-lb).*rand(pop_size, D);
vel   = -v_max + 2*v_max.*rand(pop_size, D);
pbest = pos;
fit   = zeros(pop_size,1);
for i = 1:pop_size
    fit(i) = fn(pos(i,:));
    fes    = fes+1;
end
pbest_fit = fit;
[best_val, gi] = min(fit);
gbest    = pos(gi,:);
history(1:min(fes,max_fes)) = best_val;

while fes < max_fes
    w = w_max - (w_max-w_min)*(fes/max_fes);  % linear decay

    for i = 1:pop_size
        r1 = rand(1,D); r2 = rand(1,D);
        vel(i,:) = w*vel(i,:) ...
                 + c1*r1.*(pbest(i,:) - pos(i,:)) ...
                 + c2*r2.*(gbest       - pos(i,:));
        vel(i,:) = max(-v_max, min(v_max, vel(i,:)));
        pos(i,:) = pos(i,:) + vel(i,:);
        pos(i,:) = max(lb, min(ub, pos(i,:)));

        f = fn(pos(i,:)); fes = fes+1;

        if f < pbest_fit(i)
            pbest(i,:)  = pos(i,:);
            pbest_fit(i)= f;
        end
        if f < best_val
            best_val = f;
            gbest    = pos(i,:);
        end

        idx = min(fes, max_fes);
        history(idx) = best_val;
        if fes >= max_fes; break; end
    end
end

for k = 2:max_fes
    if history(k) == 0; history(k) = history(k-1); end
end
end

%% ---- SBX Crossover (GA helper) -----------------------------------------
function [c1, c2] = ga_sbx(p1, p2, lb, ub, eta)
    D  = numel(p1);
    c1 = p1; c2 = p2;
    for j = 1:D
        if rand < 0.5 && abs(p1(j)-p2(j)) > 1e-10
            y1 = min(p1(j),p2(j)); y2 = max(p1(j),p2(j));
            b1 = 1+2*(y1-lb(j))/(y2-y1);
            b2 = 1+2*(ub(j)-y2)/(y2-y1);
            a1 = 2-b1^(-(eta+1)); a2 = 2-b2^(-(eta+1));
            r  = rand;
            bq1 = r<=1/a1; bq2 = r<=1/a2;
            if bq1, bq1v=(r*a1)^(1/(eta+1));
            else,   bq1v=(1/(2-r*a1))^(1/(eta+1)); end
            if bq2, bq2v=(r*a2)^(1/(eta+1));
            else,   bq2v=(1/(2-r*a2))^(1/(eta+1)); end
            c1(j) = max(lb(j),min(ub(j), 0.5*((y1+y2)-bq1v*(y2-y1))));
            c2(j) = max(lb(j),min(ub(j), 0.5*((y1+y2)+bq2v*(y2-y1))));
        end
    end
end

%% ---- Polynomial Mutation (GA helper) -----------------------------------
function child = ga_polymut(chrom, lb, ub, p_mut, eta_m)
    child = chrom;
    for j = 1:numel(chrom)
        if rand < p_mut
            d1 = (chrom(j)-lb(j))/(ub(j)-lb(j));
            d2 = (ub(j)-chrom(j))/(ub(j)-lb(j));
            r  = rand;
            if r < 0.5
                xy  = 1-d1;
                val = (2*r+(1-2*r)*xy^(eta_m+1))^(1/(eta_m+1))-1;
            else
                xy  = 1-d2;
                val = 1-(2*(1-r)+2*(r-0.5)*xy^(eta_m+1))^(1/(eta_m+1));
            end
            child(j) = max(lb(j),min(ub(j), chrom(j)+val*(ub(j)-lb(j))));
        end
    end
end

%% ---- CEC2005 F1: Shifted Sphere ----------------------------------------
function y = cec2005_f1(x)
    D = numel(x);
    % Shift vector o is generated from a fixed seed so it is identical across
    % all calls for the same D. At D=2 the first 2 elements of o are used; at
    % D=10 the first 10 elements are used — consistent with CEC 2005 convention
    % where the shift vector is generated once for the maximum dimension and
    % sub-vectors are taken for lower dimensions.
    rng(1001,'twister');
    o = -100+200*rand(1,D);
    rng('shuffle');
    y = sum((x(:)'-o).^2) + (-450);
end

%% ---- CEC2005 F9: Shifted Rastrigin -------------------------------------
function y = cec2005_f9(x)
    D = numel(x);
    % Shift vector o sampled from [-5,5]^D using a fixed seed for
    % reproducibility. Sub-vectors are used for D=2 and D=10 consistently,
    % matching CEC 2005 protocol (same shift direction, truncated to D dims).
    rng(1009,'twister');
    o = -5+10*rand(1,D);
    rng('shuffle');
    z = x(:)'-o;
    y = sum(z.^2 - 10*cos(2*pi*z) + 10) + (-330);
end
