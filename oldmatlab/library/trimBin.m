% Function trims a bin so that it doesn't
% go beyond zero to start
% or go beyond the length of an array to end

function [trim_start, trim_end] = trimBin(startspot, endspot, n)

    if startspot < 1
        trim_start = 1;
    else
        trim_start = startspot;
    end

    if endspot > n
        trim_end = n;
    else
        trim_end = endspot;
    end

end