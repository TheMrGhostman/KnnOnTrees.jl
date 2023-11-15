using StatsBase, Random, ArgParse, DrWatson

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        arg_type = String
        help = "Name of dataset to use"
        default = "Mutagenesis"
    "maxseed"
        arg_type = Int
        help = "number of splits "
        default = 3
    "repetitions"
        arg_type = Int
        help = "number of repetitions"
        default = 1

end

parsed_args = parse_args(ARGS, s)
@unpack dataset, maxseed, repetitions = parsed_args
# dataset, maxseed, repetitions = "Mutagenesis", 3, 1

default_iters = 200000

function sample_params()
	par_vec = ( 
		vcat(map(x->[1,3,5,7] .* x, 10f0 .^(-5:-1))...),
		2 .^ (1:4),
		#[100000], 
		1:Int(1e8)
	)
	argnames = (
		:learning_rate,
		:batch_size, 
		#:iters, 
		:init_seed
	)
	return (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
end


for rep ∈ 1:repetitions
    params = sample_params()
    for i ∈ 1:maxseed
        #mycommand = `sbatch ./../runscripts/run_triplet.sh $(dataset) $(i) $(params.iters) $(params.learning_rate) $(params.batch_size) $(params.init_seed)`
        mycommand = `sbatch ./../runscripts/run_triplet.sh $(dataset) $(i) $(default_iters/params.batch_size) $(params.learning_rate) $(params.batch_size) $(params.init_seed)`
        @info mycommand
        run(mycommand)
    end
end