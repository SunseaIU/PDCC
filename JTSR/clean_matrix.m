function M = clean_matrix(M)
    % 替换NaN和Inf为0
    M(isnan(M) | isinf(M)) = 0;
    % 如果需要，可以进一步处理
    % 例如，如果某一行或某一列全是NaN或Inf，可以删除该行或列
    % 但这里我们只是简单地替换为0
end