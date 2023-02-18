## How to run the code
#### Install the conda environment
> conda env create -n 2020101024A1 --file environment.yml

#### Q2 getting  smoothing results
> conda activate 2020101024A1   
bash generate_reports.sh

I am outputing the ngrams for which the likelyhood is 0. These cause the avg perplexity to become inf. So I am caliculating avg perplexity ignoring these ngrams sentences perplexity.

#### Q2 getting terminal
> conda activate 2020101024A1
python language_model.py w/k path_to_corpus


#### Q3 generating models
> conda activate 2020101024
python neural_language_model.py 0 'path to corpus' 0