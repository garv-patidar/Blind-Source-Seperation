function separatedSignals = blindSourceSeparation(mixedSignalFiles)
    % Blind Source Separation using FastICA for any number of input files
    % Usage: blindSourceSeparation({'mix1.wav', 'mix2.wav', ...})

    [mixedSignals, fs] = loadMixedSignals(mixedSignalFiles);
    numSignals = length(mixedSignals);

    % Prepare signals matrix (each row is one signal)
    X = zeros(numSignals, length(mixedSignals{1}));
    for i = 1:numSignals
        X(i, :) = mixedSignals{i}';
    end

    % Plot original mixed signals
    plotSignals(X, fs, 'Mixed Signal');

    % Center and scale the signals
    X_normalized = (X - mean(X, 2)) ./ std(X, 0, 2);

    % Whiten the signals
    [X_white, ~] = whiten(X_normalized);

    % Apply FastICA algorithm
    ica_model = fastICA(X_white);

    % Get separated signals
    separatedSignals = ica_model * X_white;

    % Plot separated signals
    plotSignals(separatedSignals, fs, 'Separated Signal');

    % Save separated signals to WAV files
    saveSignals(separatedSignals, fs);
end

function [mixedSignals, fs] = loadMixedSignals(filePaths)
    % Load all mixed signals from WAV files
    numSignals = length(filePaths);
    mixedSignals = cell(1, numSignals);

    [signal, fs] = audioread(filePaths{1});
    if size(signal, 2) > 1  % Convert stereo to mono
        signal = mean(signal, 2);
    end
    mixedSignals{1} = signal;

    for i = 2:numSignals
        [signal, fs_i] = audioread(filePaths{i});
        if fs_i ~= fs
            error('All files must have the same sampling rate');
        end
        if size(signal, 2) > 1
            signal = mean(signal, 2);
        end
        mixedSignals{i} = signal;
    end

    % Ensure all signals have the same length
    minLength = min(cellfun(@length, mixedSignals));
    for i = 1:numSignals
        mixedSignals{i} = mixedSignals{i}(1:minLength);
    end
end

function [X_white, V] = whiten(X)
    % Whitening transformation (ZCA whitening)
    C = X * X' / size(X, 2);  % Covariance matrix
    [E, D] = eig(C);          % Eigen decomposition
    V = E * diag(1./sqrt(diag(D) + 1e-5)) * E'; % Whitening matrix
    X_white = V * X;          % Whitened data
end

function W = fastICA(X)
    % FastICA implementation with fixed dimensions
    [n, m] = size(X);
    W = orth(randn(n, n));    % Random orthogonal initialization

    maxIter = 1000;
    tolerance = 1e-6;

    for iter = 1:maxIter
        W_old = W;

        % Update each component
        for j = 1:n
            w = W(j, :);

            % Nonlinearity (using tanh for stability)
            wx = w * X;
            g = tanh(wx);
            g_prime = 1 - g.^2;

            % Fixed-point update with proper dimensions
            w_new = (X * g')' / m - mean(g_prime) * w;

            % Orthogonalization
            for k = 1:j-1
                w_new = w_new - (w_new * W(k,:)') * W(k,:);
            end

            % Normalization
            w_new = w_new / norm(w_new);
            W(j,:) = w_new;
        end

        % Check convergence
        if norm(abs(W * W_old') - eye(n), 'fro') < tolerance
            break;
        end
    end
end

function plotSignals(X, fs, titlePrefix)
    % Plot all signals with proper time axis and scaling
    numSignals = size(X, 1);
    duration = size(X, 2) / fs;
    t = linspace(0, duration, size(X, 2));

    rows = ceil(sqrt(numSignals));
    cols = ceil(numSignals / rows);

    figure('Name', [titlePrefix 's']);
    for i = 1:numSignals
        subplot(rows, cols, i);
        plot(t, X(i, :));
        title(sprintf('%s %d', titlePrefix, i));
        xlabel('Time (s)');
        ylabel('Amplitude');
        y_range = max(abs(X(i, :))) * 1.1;
        ylim([-y_range y_range]);
        xlim([0 min(2, duration)]); % Show first 2 seconds for clarity
        grid on;
    end
    set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure
end

function saveSignals(signals, fs)
    % Save separated signals with proper normalization
    numSignals = size(signals, 1);
    for i = 1:numSignals
        signal = signals(i, :);

        % Normalize to prevent clipping
        signal = signal / max(abs(signal)) * 0.9;

        filename = sprintf('separated_signal_%d.wav', i);
        audiowrite(filename, signal', fs);
        fprintf('Saved %s\n', filename);
    end
end
